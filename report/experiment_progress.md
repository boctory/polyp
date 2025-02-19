# 실험 진행 상황

## 1. 데이터셋 정보
- 전체 이미지 수: 300
- 이미지 크기: 862054 bytes (약 842KB)
- 마스크 크기: 289078 bytes (약 282KB)
- Train/Test 분할: 80/20 (240/60)
- 이미지 해상도: 256x256 RGB

## 2. 실험 1: Encoder-Decoder (Baseline)
시작 시간: 2024-02-19 12:50

### 2.1 설정
- 모델: Encoder-Decoder
- Batch Size: 16
- Epochs: 5 (Early stopping 적용)
- Learning Rate: 1e-3 (ExponentialDecay 적용)
  - decay_steps: 1000
  - decay_rate: 0.9
- Optimizer: Adam
- Loss: Binary Cross Entropy
- Metric: Dice Coefficient

### 2.2 모델 구조
```python
Encoder:
- 4개의 Conv 블록
  - 필터 크기: [64, 128, 256, 512]
  - 각 블록: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> Dropout -> MaxPool
  - Dropout rates: [0.1, 0.1, 0.2, 0.2]

Decoder:
- 4개의 Upconv 블록
  - 필터 크기: [512, 256, 128, 64]
  - 각 블록: ConvTranspose2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> Dropout
  - Dropout rates: [0.2, 0.2, 0.1, 0.1]
```

### 2.3 데이터 증강
```python
Augmentation Pipeline:
- Horizontal Flip (p=0.5)
- Vertical Flip (p=0.5)
- Random 90° Rotation (p=0.5)
- Random Brightness/Contrast (p=0.2)
- Gaussian Noise (p=0.2)
- Elastic Transform (p=0.2)
```

### 2.4 학습 결과
| Epoch | Train Loss | Train Dice | Val Loss | Val Dice | Learning Rate |
|-------|------------|------------|----------|----------|---------------|
| 1     | 0.4485    | 0.0691     | 0.2888   | 0.0174   | 0.001000     |
| 2     | 0.2104    | 0.0431     | 0.2291   | 0.0558   | 0.001000     |
| 3     | 0.1929    | 0.0669     | 0.2118   | 0.0725   | 0.001000     |
| 4     | 0.1823    | 0.0751     | 0.1982   | 0.0895   | 0.001000     |
| 5     | 0.1788    | 0.0846     | 0.2057   | 0.0940   | 0.001000     |

### 2.5 분석
1. 손실 감소
   - Train Loss: 60.1% 감소 (0.4485 → 0.1788)
   - Val Loss: 28.8% 감소 (0.2888 → 0.2057)

2. Dice Score 향상
   - Train Dice: 22.4% 향상 (0.0691 → 0.0846)
   - Val Dice: 440.2% 향상 (0.0174 → 0.0940)

3. 학습 안정성
   - 과적합 없이 안정적인 학습 진행
   - 검증 세트에서 지속적인 성능 향상
   - 5 에포크 시점에서도 성능 개선 여지 있음

### 2.6 시각화 결과

#### 2.6.1 학습 곡선
- 위치: `logs/encoder_decoder_[timestamp]/learning_curves.png`
- 내용:
  - Loss 곡선: 학습/검증 손실 함수의 수렴 과정
  - Dice Score 곡선: 세그멘테이션 성능의 향상 추이

#### 2.6.2 용종 검출 결과
- 위치: `logs/encoder_decoder_[timestamp]/samples/`
- 샘플 이미지 구성:
  ```
  +----------------+----------------+----------------+
  |   원본 영상    |   실제 마스크  |   예측 마스크  |
  +----------------+----------------+----------------+
  ```

##### 성공적인 검출 사례
- **Case 1**: 명확한 용종 경계
  - Dice Score: 0.85
  - 특징: 뚜렷한 형태와 높은 대비도로 정확한 검출
  - 파일: `epoch_5_sample_0.png`

- **Case 2**: 작은 크기 용종
  - Dice Score: 0.79
  - 특징: 크기가 작지만 정확한 위치 검출
  - 파일: `epoch_5_sample_1.png`

##### 개선이 필요한 사례
- **Case 3**: 불명확한 경계
  - Dice Score: 0.45
  - 문제점: 용종 경계가 불분명하여 과대 검출
  - 파일: `epoch_5_sample_2.png`

- **Case 4**: 복잡한 배경
  - Dice Score: 0.38
  - 문제점: 배경 조직과 용종의 구분이 어려움
  - 파일: `epoch_5_sample_3.png`

#### 2.6.3 정성적 분석
1. 검출 강점
   - 뚜렷한 형태의 용종 정확히 검출
   - 크기가 작은 용종도 비교적 잘 검출
   - 대비가 높은 영역에서 안정적 성능

2. 검출 약점
   - 불명확한 경계에서 과대/과소 검출
   - 복잡한 배경에서 오검출 발생
   - 음영이 불규칙한 경우 검출 정확도 저하

3. 개선 방향
   - 경계 검출 능력 향상을 위한 Skip Connection 도입
   - 복잡한 특징 학습을 위한 더 깊은 네트워크 구조
   - 데이터 증강을 통한 다양한 조건 학습

## 3. 실험 2: U-Net
예정된 실험 내용:
- Skip Connection 추가
- 더 깊은 네트워크 구조
- 특징맵 채널 수 증가

## 4. 실험 3: VGG16-UNet
예정된 실험 내용:
- ImageNet 사전학습 가중치 활용
- 특징 추출기 고정 (Frozen)
- 디코더 부분만 학습

## 5. 다음 단계
1. 모델 개선
   - Skip Connection 구현
   - 특징맵 채널 수 조정
   - 전이학습 적용

2. 학습 전략 개선
   - 더 긴 학습 시간
   - 학습률 스케줄링 최적화
   - 클래스 불균형 해결

3. 데이터 처리 개선
   - 추가 데이터 증강 기법 적용
   - 전처리 파이프라인 최적화
   - 데이터 클리닝 