## NPU 기반 멀티모달 모델을 활용한 시각장애인 보행 보호 어플리케이션 구현 및 성능 최적화


본 프로젝트는 NPU 환경에서 멀티모달 모델을 실행하여 시각장애인의 보행 안전을 지원하는 어플리케이션이다. 카메라 입력으로부터 실시간 영상 정보를 분석하거나 음성(Speech-to-Text)으로 받은 사용자 질문에 응답을 생성하고, 이를 음성(Text-to-Speech)으로 변환하여 제공한다. 


Rebellions ATOM+ CA22 NPU를 기반으로 멀티모달 모델을 컴파일 및 실행하며, Gradio 기반 웹 인터페이스를 통해 데스크톱 및 모바일 환경에서 테스트할 수 있다. 


```text
├── model/                            # NPU compile을 위한 모델 파일
│   ├── __init__.py
│   ├── configuration_llava_ov.py
│   └── modeling_llava_ov.py
│
├── compile.py                        # NPU compile 스크립트
├── run.py                            # 최종 시스템 실행 스크립트
└── README.md
```


###  Run Command
```bash
# Rebellions SDK 사전 설치 필요
python3 compile.py
python3 run.py
# 실행 완료 시, 터미널에 Gradio 링크가 출력되며
# 해당 링크를 통해 어플리케이션 테스트 가능 (모바일 Chrome 환경 권장)
```


###  Reference
- [LLaVA-Onevision (paper): Multimodal Model used in this system] (https://arxiv.org/abs/2408.03326)
- [Rebellions SDK] (https://github.com/rebellions-sw/optimum-rbln)
- [RealtimeTTS: Open-source Text-to-Speech library] (https://github.com/KoljaB/RealtimeTTS/tree/master)
