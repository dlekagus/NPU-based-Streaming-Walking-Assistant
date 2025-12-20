import asyncio
import gradio as gr
import numpy as np
import torch
import time
import cv2
from transformers import AutoProcessor, pipeline, AutoTokenizer
from optimum.rbln import RBLNLlavaOnevisionForConditionalGeneration

from RealtimeTTS import TextToAudioStream, SystemEngine, GTTSEngine
import soundfile as sf
import tempfile
import os

# 외부 경량 TTS 모델 로딩
from transformers import pipeline as hf_pipeline

# 성능 최적화: frame selection
def selection_difffp(frames, k=16, resize=32):
    if len(frames) <= k:
        return frames

    # resizing, grayscale
    def preprocess(frame):
        f = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_LINEAR)
        f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        return f.astype(np.float32).flatten() / 255.0

    F = [preprocess(f) for f in frames]

    # 1-norm distance 
    F_dists = []
    for i in range(1, len(F)):
        F_dist = np.abs(F[i] - F[i-1]).sum()
        F_dists.append(F_dist)

    # top-16
    topk_F = np.argpartition(F_dists, - (k-1))[-(k-1):]
    topk_F = np.sort(topk_F + 1)
    
    selected_F_idx = [0] + topk_F.tolist()
    selected_F = [frames[i] for i in selected_F_idx]

    return selected_F

# 추론 stage별 latency 측정: time wrapper
class TimeWrapper:
    def __init__(self, wrapped, name="Module", loop_wrapped=None):
        self._wrapped = wrapped
        self.name = name
        self.times = []
        self._loop_wrapped = loop_wrapped
    
    def __call__(self, *args, **kwargs):
        start = time.perf_counter_ns()
        output = self._wrapped(*args, **kwargs)
        end = time.perf_counter_ns()
        elapsed = 0.000000001 * (end - start)
        self.times.append(elapsed)
        return output

# --- 모델 로딩 (Llava) ---
model_dir = "/home/work/npu2/optimum-rbln/src/compiled-llava-ov-7b-chat-hf-batch4/" # 컴파일한 llava-ov 파일 경로 지정
processor = AutoProcessor.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = RBLNLlavaOnevisionForConditionalGeneration.from_pretrained(
    model_dir,
    export=False,
    rbln_config={
        "device": 0,
        "vision_tower": {},
        "language_model": {
            "tensor_parallel_size": 1,
            "use_inputs_embeds": True,
        },
    },
)

# TimeWrapper로 각 모듈 감싸서 VLM 내 각 모듈 시간 측정
model.vision_tower = TimeWrapper(model.vision_tower, name="ViT")
model.multi_modal_projector = TimeWrapper(model.multi_modal_projector, name="MLPprojector")
model.language_model.prefill_decoder = TimeWrapper(model.language_model.prefill_decoder, name="Prefill")
model.language_model.decoder = TimeWrapper(model.language_model.decoder, name="Decode")

# speech to text (Whisper)
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")


# test to speech (RealtimeTTS)
tts_engine = GTTSEngine()
tts_stream = TextToAudioStream(engine=tts_engine, muted=True)


# --- 글로벌 상태 ---
frame_buffer = []
NUM_FRAMES_TO_COLLECT = 64
prompt = ""
is_QA_start = False
is_QA_end = False
e2e_start_ns = None
difffp_time = []
TTS_time = []
end2end_time =[]


# ---  음성 인식 함수 ---
def transcribe_audio(audio_filepath): # 음성 파일을 Whisper STT 모델로 텍스트로 변환
    global prompt, is_QA_end
    try:
        #print(f"[INFO] 음성 인식 시작: {audio_filepath}")
        start_stt = time.perf_counter_ns()
        transcription = stt_pipeline(audio_filepath, generate_kwargs={"task": "translate"}) # 오디오 파일을 입력으로 받아 텍스트로 변환
        text = transcription.get('text', '')
        prompt = text # 전역 prompt 변수에 저장
        end_stt = time.perf_counter_ns()
        print(f"STT time: {0.000000001*(end_stt - start_stt)} s")
        #print(f"[INFO] 음성 인식 결과: {text}")
        is_QA_end = True # QA 종료 플래그(is_QA_end)를 True로 설정
        return text
    except Exception as e:  # 모든 예외를 받아 처리
        #print(f"[INFO] 음성 인식 대기중: {e}")
        prompt = ""          # 필요 시 전역 변수도 초기화
        return "음성 인식 대기중"  


# --- 추론 함수 ---
def run_inference(frames, prompt, num_new_tokens=30):

    #print(f"[INFO] {len(frames)}개 프레임과 프롬프트 '{prompt}'로 추론을 시작합니다.")
    
    content = [{"type": "video"}]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=text,
        videos=[frames],
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=num_new_tokens, 
            # eos_token_id='.', '.' id : 13 
        )
    input_token_len = inputs.input_ids.shape[1]
    generated_ids = output_ids[:, input_token_len:]
    english_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"ViT: {model.vision_tower.times[-1:]}")
    print(f"projector: {model.multi_modal_projector.times[-1:]}")
    print(f"prefill: {model.language_model.prefill_decoder.times[-1:]}")
    print(f"decode: {sum(model.language_model.decoder.times[-num_new_tokens+1:])}")
    #print(f"[INFO] 추론 결과: {english_output}")
    return english_output


# --- TTS 함수 ---
def tts_transform(text): # 텍스트를 RealtimeTTS와 GTTSEngine을 이용해 음성으로 변환
    global e2e_start_ns, TTS_time, end2end_time
    start_TTS = time.perf_counter_ns()
    if not text or not text.strip():
        return None, None
    
    if is_QA_start and not is_QA_end:
        return None, None

    # 모델의 답변에서 옵션 부분을 제거하여 자연스러운 문장만 남김
    # 예: "(A) continue straight." -> "continue straight."
    cleaned_text = text.split('→')[-1].strip().split('Answer:')[-1].strip()
    # 괄호와 마침표 제거
    cleaned_text = cleaned_text.replace('(', '').replace(')', '').replace('.', '')
    if not cleaned_text.endswith('.'):
        cleaned_text += '.'
    #print(f"[INFO] TTS 변환 시작: '{cleaned_text}'")

    # 임시로 wav 파일 생성
    fd, out_wav = tempfile.mkstemp(suffix=".wav", prefix="tts_")
    os.close(fd)
    
    tts_stream.reset_generated_text = True # 이전 TTS 요청에서 남아 있을 수 있는 텍스트 상태를 초기화
    tts_stream.feed(cleaned_text) # TTS 엔진에 변환할 텍스트를 입력
    tts_stream.play(output_wavfile=out_wav, buffer_threshold_seconds=0.2) # TTS 변환 후 생성된 음성을 임시 wav 파일로 저장

    data, sr = sf.read(out_wav, dtype='float32') # 생성된 wav 파일을 로드
    os.remove(out_wav) # 임시 파일은 더 이상 필요 없으므로 삭제

    if is_QA_start and not is_QA_end:
        return None, None
    
    #print(f"[INFO] 변환 완료: '{cleaned_text}'")
    end_TTS = time.perf_counter_ns()
    TTS_time.append(0.000000001 * (end_TTS - start_TTS))
    print(f"TTS time: {round(float(np.mean(TTS_time[1:])), 6)} s")
    
    if e2e_start_ns is not None:
        end2end_time.append(0.000000001 * (end_TTS - e2e_start_ns))
        print(f"end2end time: {round(float(np.mean(end2end_time[1:])), 6)} s")
        e2e_start_ns = None

    return (sr, data), text

def collect_frame(frame): # 실시간으로 웹캠/스마트폰에서 입력되는 프레임 resize 후 buffer에 저장
    global frame_buffer

    # 입력 프레임 저장된 크기로 resize
    target_size = (480, 360)
    frame_buffer.append(cv2.resize(frame, target_size))

    # Gradio UI에 현재 버퍼 상태 표시 위한 문자열
    status_text = f"버퍼 상태: {len(frame_buffer)}/{NUM_FRAMES_TO_COLLECT} 프레임"

    # 버퍼가 최대 프레임 수를 초과한 경우 가장 오래된 프레임 제거, 최근 프레임만 유지
    if len(frame_buffer) >= NUM_FRAMES_TO_COLLECT: 
        frame_buffer = frame_buffer[1:]

    return status_text


def QA(frame): # QA 모드, 사용자의 음성 질문과 최근 1분동안의 프레임 기반 추론
    # 전역 prompt 사용, STT 결과로 저장된 사용자 질문 포함
    global prompt
    prompt = f"In brief, {prompt}."
    # frame selection
    start_difffp = time.perf_counter_ns()
    selected_frames = selection_difffp(frame, k=16)
    end_difffp = time.perf_counter_ns()
    difffp_time.append(0.000000001 * (end_difffp - start_difffp))
    print(f"difffp time: {round(float(np.mean(difffp_time[1:])), 6)} s")
    #return run_inference(selected_frames, prompt)
    start_QA = time.perf_counter_ns()
    #result = run_inference(frame, prompt)
    result = run_inference(selected_frames, prompt)
    end_QA = time.perf_counter_ns()
    #print(f"QA inference time: {0.000000001*(end_QA - start_QA)} s")
    
    return result.split('.')[0] + '.'
    prompt = ""

def reminder(frame): # reminder 모드, 최근 4 프레임을 바탕으로 2-stage prompting 적용, 간결한 답변 생성
    reminder_prompt = "describe this video"

    start_reminder = time.perf_counter_ns()
    description = run_inference(frame, reminder_prompt, 128) # 추론 1, VLM을 사용하여 현 상황 자세한 묘사 생성
    end_reminder = time.perf_counter_ns()

    # 2단계 요약 프롬프트, {text_input}에 추론 1에서 생성한 텍스트 삽입
    summarization_prompt = """You are a walking guide for a visually impaired person.
                            Based ONLY on the following description, extract the most critical information for safe walking in one very brief sentence (under 15 words). 
                            Focus on obstacles or path conditions directly ahead.
                            Please consider the user can't see any signs.
                            
                            Description: "{text_input}"
                            
                            Now, provide the critical navigation summary.
                            """
    # 추론 2, 추론 1에서 생성한 자세한 묘사를 바탕으로 LLM을 사용하여 현 상황을 간결하게 요약하여 표현하는 답변 생성
    summary_prompt = summarization_prompt.format(text_input=description)
    summary = run_inference(prompt=summary_prompt)

    print(f"묘사: {description}")
    print(f"요약: {summary}")
    #print(f"reminder inference time: {0.000000001*(end_reminder - start_reminder)} s")
    
    return summary

def QA_reminder(): # 사용자 질문 여부에 따라 QA와 reminder 모드 결정
    global frame_buffer, is_QA_start, is_QA_end, e2e_start_ns
    e2e_start_ns = time.perf_counter_ns()
    if is_QA_start:
        while not is_QA_end:
            time.sleep(0.2)
        is_QA_start=False
        is_QA_end=False
        return QA(frame_buffer)
    else:
        return reminder(frame_buffer[-4:]) # reminder 모드 사용 시 버퍼에서 최근 4 프레임만 추출해 입력
    #return QA(frame_buffer)
    
def set_is_QA_start():
    global is_QA_start
    is_QA_start = True

def set_is_QA_end():
    global is_QA_end
    is_QA_end = True
    
# --- Gradio UI ---
# 실시간 웹캠 영상 + 음성 질문 기반 멀티모달 QA 인터페이스
# Start / Stop 버튼으로 추론 루프 제어
with gr.Blocks() as demo:
    # UI 타이틀
    gr.Markdown("실시간 비디오 QA")

    # 메인 레이아웃 (좌: 입력 / 우: 출력)
    with gr.Row():
        # 좌측: 입력 영역
        with gr.Column(scale=2):
            # 실시간 웹캠 입력
            input_img = gr.Image(sources=['webcam'], type='numpy', label="실시간 웹캠", mirror_webcam=False)
            # 마이크 음성 입력 (질문용)
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="음성으로 질문하기")
            # 추론 시작 / 중지 버튼
            start_btn = gr.Button("▶️ Start Inference")
            stop_btn = gr.Button("⏸️ Stop Inference")

        # 우측: 출력 및 상태 표시 영역
        with gr.Column(scale=3):
            # STT 결과 표시
            prompt_text = gr.Textbox(
                label="인식된 질문 (Prompt)", 
                value="", 
                placeholder="음성으로 질문하거나 여기에 직접 입력하세요...",
                interactive=True, 
                lines=2
            )
            # 모델의 텍스트 응답 표시
            display_text = gr.Textbox(label="모델 응답", value="모델 초기화 완료. 질문을 입력하세요.", interactive=False, lines=8)
            # 프레임 버퍼 상태 표시
            status_text = gr.Textbox(label="버퍼 상태", value="수집 대기 중...", interactive=False)
            # TTS 음성 출력 (자동 재생)
            audio_output = gr.Audio(label="음성 답변", autoplay=True)

    # 모델 출력 임시 저장 상태 변수
    output_text = gr.State()

    # 음성 녹음 시작 시 QA 모드 플래그 활성화
    audio_input.start_recording(set_is_QA_start,None,None)
    
    # 음성 입력이 완료되면 STT 수행 → prompt_text에 반영
    audio_input.change(
        fn=transcribe_audio, 
        inputs=audio_input, 
        outputs=prompt_text
    )

    # 웹캠 프레임 스트리밍
    input_img.stream(
        fn=collect_frame, 
        inputs=[input_img], 
        outputs=[status_text],
        time_limit=0.05, 
        stream_every=0.01, 
        concurrency_limit=30
    )

    def start_timer():
        # 추론 루프 시작
        gr.Info("▶️ Start Inference")
        timer = gr.Timer(20, active=True)
        return timer

    def stop_timer():
        # 추론 루프 중지
        gr.Info("⏸️ Stop Inference")
        timer = gr.Timer(20, active=False)
        return timer

    timer = gr.Timer(1, active=False)

    # Start 버튼 동작, 타이머 활성화 -> 추론 실행 -> TTS 변환 후 출력 
    start_btn.click(start_timer, outputs=timer).then(
        fn=QA_reminder,
        outputs = output_text
    ).then(
        fn=tts_transform,
        inputs=output_text,
        outputs=[audio_output, display_text]
    )

    # Stop 버튼 → 타이머 비활성화
    stop_btn.click(stop_timer, outputs=timer)

    # 타이머 기반 자동 추론 루프
    timer.tick(
        fn=QA_reminder, # VLM 모델로 답변 추론 
        outputs = output_text
    ).then(
        fn=tts_transform,
        inputs=output_text, # 추론 결과 TTS로 출력
        outputs=[audio_output, display_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)  
