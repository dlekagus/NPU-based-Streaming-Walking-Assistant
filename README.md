## NPU 기반 멀티모달 모델을 활용한 시각장애인 보행 보호 어플리케이션 구현 및 성능 최적화


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


### Execution Enviroment
- NPU: 리벨리온(Rebellions) ATOM+ CA22
- Gradio 기반 웹 인터페이스 (브라우저 접근으로 모바일 환경 사용 가능)


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
- [RealtimeTTS: Open-source Text-to-Speech library] (https://github.com/KoljaB/RealtimeTTS/tree/master)
