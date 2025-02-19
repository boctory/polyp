# 내시경 영상에서의 용종 검출 프로젝트

## 프로젝트 소개
본 프로젝트는 내시경 영상에서 용종을 자동으로 검출하는 딥러닝 기반 세그멘테이션 모델을 개발하는 것을 목표로 합니다. 
GIANA (Gastrointestinal Image ANAlysis) 데이터셋을 활용하여 다양한 딥러닝 아키텍처를 실험하고 비교 분석했습니다.

### 개발 목표
1. 정확한 용종 영역 검출
2. 실시간 처리가 가능한 효율적인 모델 구현
3. 다양한 내시경 환경에서의 강건한 성능

## 개발 스토리

### 1. 프로젝트 시작 (개발 환경 설정)
1. **개발 환경 구축**
   ```bash
   # 1. 가상환경 생성 및 활성화
   python -m venv venv
   source venv/bin/activate

   # 2. 필요한 라이브러리 설치
   pip install tensorflow numpy opencv-python albumentations
   pip install matplotlib tqdm pillow scikit-learn

   # 3. 의존성 저장
   pip freeze > requirements.txt
   ```

2. **프로젝트 구조 설계**
   - 모듈화된 구조로 설계하여 유지보수성 고려
   - 실험 결과와 소스코드 분리하여 관리

### 2. 데이터 처리 파이프라인 구현
1. **데이터 전처리 (data_loader.py)**
   ```python
   # 1. 이미지 로딩 및 전처리
   def _load_image(self, image_path, is_mask=False):
       img = tf.io.read_file(image_path)
       img = tf.image.decode_bmp(img_raw, channels=3)
       img = tf.image.resize(img, self.img_size)
       return img

   # 2. 데이터 증강 파이프라인 구성
   self.aug_pipeline = A.Compose([
       A.HorizontalFlip(p=0.5),
       A.VerticalFlip(p=0.5),
       A.RandomRotate90(p=0.5),
       # ... 추가 증강 기법
   ])
   ```

### 3. 모델 아키텍처 구현 (models.py)
1. **기본 Encoder-Decoder 모델**
   - 처음에는 단순한 구조로 시작
   - Conv2D와 BatchNorm 위주로 구현

2. **U-Net 구현 시 겪은 어려움**
   ```python
   # Skip Connection 구현에서 어려웠던 점
   def call(self, x):
       skip_connections = []
       for encoder_block in self.encoder:
           x = encoder_block(x)
           skip_connections.append(x)  # 특징맵 저장
       
       # 디코더에서 특징맵 결합 시 차원 맞추기가 까다로웠음
       skip_connections = skip_connections[::-1]
       for decoder_block, skip in zip(self.decoder, skip_connections):
           x = decoder_block(x)
           x = layers.Concatenate()([x, skip])
   ```

### 4. 학습 과정에서의 시행착오
1. **메모리 관리**
   - 처음에는 OOM (Out of Memory) 오류 발생
   - 배치 크기 조정과 tf.data.Dataset의 prefetch 활용으로 해결

2. **학습 안정성 확보**
   ```python
   # 1. 학습률 스케줄링 도입
   initial_learning_rate = 1e-3
   lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
       initial_learning_rate, decay_steps=1000, decay_rate=0.9
   )

   # 2. Early Stopping 구현
   patience_counter = 0
   best_val_dice = 0
   ```

### 5. 디버깅 및 문제 해결
1. **주요 문제점과 해결 방법**
   - 낮은 Dice Score
     → 데이터 증강 기법 추가 및 손실 함수 조정
   - 느린 학습 속도
     → tf.data 파이프라인 최적화
   - 과적합 문제
     → Dropout과 BatchNorm 레이어 추가

2. **성능 개선 과정**
   ```python
   # 1. 메트릭 모니터링 추가
   with summary_writer.as_default():
       tf.summary.scalar('train_loss', train_loss_metric.result())
       tf.summary.scalar('val_dice', val_dice_metric.result())

   # 2. 시각화 도구 구현
   def save_prediction_samples(model, dataset, epoch):
       # 예측 결과 시각화 및 저장
   ```

### 6. 배운 점과 성장
1. **기술적 성장**
   - TensorFlow/Keras 프레임워크 숙달
   - 딥러닝 모델 설계 및 구현 능력 향상
   - 데이터 파이프라인 최적화 경험

2. **프로젝트 관리 스킬**
   - 실험 결과 문서화의 중요성
   - 버전 관리 시스템 활용
   - 모듈화된 코드 설계의 이점

3. **향후 개선 계획**
   - Attention 메커니즘 도입 검토
   - 모델 경량화 연구
   - 하이퍼파라미터 자동 튜닝 구현

## 개발 과정

### 1. 데이터셋 구성
- GIANA Challenge 데이터셋 활용
- 전체 300장의 내시경 이미지와 마스크
- Train/Test 분할 (80:20)
- 이미지 크기: 256x256 RGB

### 2. 모델 아키텍처
세 가지 주요 모델 구현 및 비교:

1. **기본 Encoder-Decoder**
   - Baseline 모델
   - 4단계 인코더-디코더 구조
   - Dropout과 BatchNormalization 적용

2. **U-Net**
   - Skip Connection을 통한 특징 보존
   - 4단계 Contracting/Expanding 경로
   - 더 정교한 세그멘테이션 가능

3. **VGG16-UNet**
   - 전이학습 적용
   - ImageNet 사전학습 가중치 활용
   - 더 강력한 특징 추출 능력

### 3. 학습 전략
- 데이터 증강 기법 적용
  - 회전, 반전, 밝기 조정 등
  - 과적합 방지 및 일반화 성능 향상
- 학습률 스케줄링
- Early Stopping 구현
- Dice Coefficient 기반 성능 평가

## 실험 결과

### 1. Encoder-Decoder (Baseline) 성능
- Train Loss: 0.4485 → 0.1788 (60.1% 감소)
- Val Loss: 0.2888 → 0.2057 (28.8% 감소)
- Train Dice: 0.0691 → 0.0846 (22.4% 향상)
- Val Dice: 0.0174 → 0.0940 (440.2% 향상)

### 2. 주요 발견
1. Skip Connection의 중요성
2. 데이터 증강의 효과
3. 전이학습의 이점

## 프로젝트 회고

### 1. 잘된 점
- 체계적인 실험 설계와 진행
- 다양한 모델 아키텍처 비교 분석
- 안정적인 학습 파이프라인 구축

### 2. 개선 필요 사항
- 더 높은 Dice Score 달성 필요
- 클래스 불균형 문제 해결
- 실시간 처리 속도 개선

### 3. 향후 계획
- 앙상블 기법 도입
- 추가 데이터 수집
- 모델 경량화 연구

## 설치 및 실행

### 환경 설정
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 학습 실행
```bash
python train.py --data_dir data --model_type [encoder_decoder|unet|vgg16_unet] --epochs 50 --batch_size 16 --img_size 256
```

### 결과 확인
```bash
tensorboard --logdir logs
```

## 프로젝트 구조

### 주요 파일 설명

| 파일명 | 목적 | 주요 기능 |
|--------|------|-----------|
| `data_loader.py` | 데이터 처리 | • GIANA 데이터셋 로딩<br>• 이미지/마스크 전처리<br>• 데이터 증강 파이프라인<br>• Train/Test 분할 |
| `models.py` | 모델 아키텍처 | • Encoder-Decoder 구현<br>• U-Net 구현<br>• VGG16-UNet 구현<br>• 각 모델의 레이어 구성 |
| `train.py` | 학습 프로세스 | • 모델 학습 로직<br>• 손실 함수 및 메트릭 계산<br>• 학습률 스케줄링<br>• 체크포인트 저장 |
| `plot_results.py` | 결과 시각화 | • 학습 곡선 플로팅<br>• 메트릭 시각화<br>• 예측 결과 시각화 |
| `run_experiments.sh` | 실험 자동화 | • 다양한 모델 연속 학습<br>• 실험 설정 자동화<br>• 로그 관리 |

### 보고서 파일 설명

| 파일명 | 목적 | 내용 |
|--------|------|------|
| `experiment_setup.md` | 실험 설정 문서화 | • 데이터셋 구성<br>• 모델 아키텍처 설정<br>• 하이퍼파라미터 설정 |
| `experiment_progress.md` | 실험 진행 기록 | • 실험 진행 상황<br>• 중간 결과 분석<br>• 문제점 및 개선사항 |
| `experiment_results.md` | 최종 결과 정리 | • 정량적 성능 평가<br>• 정성적 분석<br>• 시각화 결과 |

### 버전 관리 제외 파일 (.gitignore)
```
# Python 관련
__pycache__/          # 파이썬 캐시 파일
*.py[cod]            # 컴파일된 파이썬 파일
venv/                # 가상 환경

# 데이터 및 로그
logs/                # 학습 로그 및 체크포인트
data/                # 데이터셋 디렉토리
*.log                # 로그 파일

# 시스템 파일
.DS_Store            # macOS 시스템 파일
.idea/               # PyCharm 설정
.vscode/             # VS Code 설정
```

### 디렉토리 구조
```
project/
├── data/              # 데이터셋 저장
│   ├── images/        # 원본 내시경 이미지
│   └── masks/         # 용종 마스크
├── logs/              # 실험 결과 저장
│   └── [model]_[timestamp]/
│       ├── samples/   # 예측 결과 시각화
│       └── metrics/   # 학습 메트릭
├── report/            # 실험 문서
├── models.py          # 모델 구현
├── data_loader.py     # 데이터 처리
├── train.py           # 학습 스크립트
├── plot_results.py    # 결과 시각화
├── run_experiments.sh # 실험 실행
└── requirements.txt   # 의존성 패키지
```

## 라이선스
MIT License

## 참고문헌
1. GIANA Challenge: https://giana.grand-challenge.org
2. U-Net 논문: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
3. VGG16 논문: "Very Deep Convolutional Networks for Large-Scale Image Recognition" 