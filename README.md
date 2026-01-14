## 🛡️ DINOv3-based Universal Deepfake Detection

본 프로젝트는 최신 비전 트랜스포머인 **DINOv3 (Vision Transformer)**를 활용하여, 다양한 위조 기법과 환경에서도 높은 일반화(Generalization) 성능을 유지하는 **범용 딥페이크 탐지 시스템**을 구축하는 것을 목표로 합니다. 

대규모 데이터셋(약 3TB 이상)의 효율적인 관리와 학습 유연성을 확보하기 위해, 크롭된 이미지를 직접 저장하지 않고 **JSON 메타데이터 기반의 실시간(On-the-fly) 전처리 파이프라인**을 채택하였습니다.

---

## 📊 Supported Datasets
모델의 강건성을 확보하기 위해 인종, 화질, 위조 방식이 각기 다른 5가지 이상의 대규모 벤치마크 데이터셋을 통합하여 활용합니다.

| Dataset | Type | Description |
| :--- | :--- | :--- |
| **FaceForensics++ (C23)** | Video | 6가지 표준 위조 기법 (Deepfakes, FaceSwap, FaceShifter 등) |
| **Celeb-DF (v2)** | Video | 고화질 얼굴 교체 및 정교한 합성 흔적 탐지 |
| **WildDeepfake** | Image | 실제 인터넷 환경의 노이즈와 다양한 배경 대응 |
| **GenImage** | Image | 최신 Diffusion 기반 생성형 AI 위조 이미지 대응 |
| **KoDF** | Video | 한국인 안면 데이터를 통한 인종적 편향성(Bias) 해소 |

---

## ⚙️ Core Technology
* **Backbone:** DINOv3 (Vision Transformer)
* **Face Detection:** InsightFace (RetinaFace / Buffalo_L)
* **Video Loading:** Decord (Fast Random Access & GPU Decoding)
* **Distributed Training:** Hugging Face `accelerate` (Multi-GPU 지원)

---

## 🚀 Key Features

### 1. JSON Metadata-driven Pipeline
2.7TB에 달하는 KoDF 등 대용량 데이터를 처리하기 위해 원본 파일에서 추출한 **얼굴 좌표 및 5-point 랜드마크 정보를 JSON으로 관리**합니다.
* **저장 공간 절약:** 크롭 이미지 저장 방식 대비 저장 공간 90% 이상 절감.
* **동적 전처리:** 학습 시점에 Margin(1.2x ~ 2.0x) 및 Face Alignment 설정을 실시간으로 변경하여 실험 가능.

### 2. Balanced Multi-Dataset Loader
데이터셋 간 불균형과 FF++의 1:5(Real:Fake) 비율 문제를 해결하기 위해 전략적인 데이터 샘플링을 수행합니다.
* **Epoch-wise Random Sampling:** FF++의 5가지 위조 기법 중 매 에폭마다 1개를 랜덤 선택하여 Real/Fake 1:1 비율 유지.
* **Identity-Leakage Prevention:** 영상 ID 기반 분할(Split)을 통해 학습 데이터 인물의 정체성이 평가 데이터에 노출되는 것을 방지.

### 3. Robust Augmentation Strategy
실제 환경의 화질 저하 및 압축 흔적을 시뮬레이션하기 위한 전처리 로직을 적용합니다.
* **Random JPEG Compression:** Quality 30~80 범위의 압축 노이즈 학습.
* **Gaussian Blur & Color Jitter:** 경계선 뭉개짐 및 조명 변화에 대한 강건성 확보.

---

## 📂 Project Structure
```text
├── tools/
│   └── data_preprocess.py     # 영상/이미지 통합 얼굴 탐지 및 JSON 메타데이터 생성
├── data/
│   ├── dataset.py             # JSON 기반 On-the-fly 크롭 및 1:1 밸런싱 데이터셋
│   └── transforms.py          # JPEG Compression 등 딥페이크 특화 Augmentation
├── models/
│   └── dinov3_detector.py     # DINOv3 기반 Binary Classifier
├── configs/
│   └── base_config.yaml       # 하이퍼파라미터 및 경로 설정
├── train.py                   # Accelerator 기반 분산 학습 메인 스크립트
└── test.py                    # 벤치마크 데이터셋 성능 평가 스크립트


--- 

# 📝 Usage
### 1. 환경 구축
```
pip install torch torchvision torchaudio
pip install insightface decord accelerate tqdm opencv-python
```

### 2. 데이터 전처리 (JSON 생성)
모든 데이터셋을 스캔하여 얼굴 좌표 및 랜드마크를 생성. (GPU 기반 병렬 처리 지원)
```
python data_preproces.py
```

### 3. 학습 시작
accelerate 설정을 마친 후 멀티 GPU 환경에서 학습을 최적화한다.
```
accelerate launch train.py
```

---
# 💡 Roadmap

- [ ] SBI (Self-Blended Images) 기법 도입을 통한 미학습 도메인 일반화 강화.
- [ ] Frequency Domain Layer: 주파수 아티팩트 탐지를 위한 DCT 분석 모듈 추가.