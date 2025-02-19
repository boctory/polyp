# 실험 설정 및 환경

## 1. 데이터셋 구성
- 데이터셋: GIANA (Gastrointestinal Image ANAlysis) Challenge
- 이미지 크기: 256 x 256
- 채널: RGB (3채널)
- Train/Test 분할: 80%/20%

## 2. 데이터 증강 기법
```python
Augmentation Pipeline:
- Horizontal Flip (p=0.5)
- Vertical Flip (p=0.5)
- Random 90° Rotation (p=0.5)
- Random Brightness/Contrast (p=0.2)
- Gaussian Noise (p=0.2)
- Elastic Transform (p=0.2)
```

## 3. 모델 아키텍처

### 3.1 Encoder-Decoder (Baseline)
- Encoder: 4개의 Conv 블록 (64, 128, 256, 512 필터)
- Decoder: 4개의 Upconv 블록
- Skip Connection 없음

### 3.2 U-Net
- Encoder: 4개의 Double Conv 블록 (64, 128, 256, 512 필터)
- Decoder: 4개의 Upconv 블록
- Skip Connection: 각 레벨별 연결

### 3.3 VGG16-UNet
- Encoder: Pretrained VGG16
- Decoder: 4개의 Upconv 블록
- Skip Connection: VGG16 중간층 특징맵 연결
- Encoder 가중치 고정 (frozen)

## 4. 학습 설정
- Batch Size: 16
- Epochs: 50
- Optimizer: Adam (lr=1e-4)
- Loss Function: Binary Cross Entropy
- Metric: Dice Coefficient
- Early Stopping: Best Validation Dice Score 기준 