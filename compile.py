import os
from optimum.rbln import RBLNLlavaOnevisionForConditionalGeneration

# Hugging Face 형식의 사전 학습된 LLaVA-OneVision 모델 경로
pretrained_model = "/home/work/npu2/optimum-rbln/src/llava-onevision-qwen2-7b-ov-chat-hf"

# RBLN NPU용 모델 로딩 및 컴파일
model = RBLNLlavaOnevisionForConditionalGeneration.from_pretrained(
    pretrained_model,
    export=True, 
    rbln_config={
        "device": [0],
        "batch_size": 4, # frame batching(성능 최적화) 위해 ViT의 batch size 4로 설정
        "vision_tower": {
            "batch_size": 4,
        },
        "language_model": {
            "tensor_parallel_size": 1,
            "max_seq_len": 32768,
            "use_inputs_embeds": True,
            "batch_size": 1,
            "prefill_chunk_size": 128,
        },
    },
    rbln_npu="RBLN-CA22",
)

# 컴파일된 모델 저장
model.save_pretrained("compiled-llava-ov-7b-chat-hf-batch4")
