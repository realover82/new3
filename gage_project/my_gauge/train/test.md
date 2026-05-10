# 다이얼 게이지 딥러닝 측정 시스템
## 초보자용 따라하기 교재
### (참고 논문: 김영훈, 석사학위논문, 2026)

---

> **이 교재는 이렇게 따라하세요**
> - 각 절의 내용을 순서대로 따라하면 됩니다
> - `회색 박스` 안의 내용은 **직접 입력하는 명령어**입니다
> - ⚠ 표시는 **주의사항**입니다
> - 💡 표시는 **알아두면 좋은 설명**입니다

---

# 제1장. 환경 구축

## 1절. 파이썬 설치 (Windows 10 기준)

### 1-1. 파이썬 다운로드

1. 웹 브라우저에서 **https://www.python.org** 로 이동합니다
2. 상단 메뉴 **Downloads** → **Python 3.12.x** 클릭
3. 설치 파일(`.exe`) 다운로드

### 1-2. 파이썬 설치

1. 다운로드된 `python-3.12.x-amd64.exe` 를 **더블클릭**합니다
2. 설치 화면에서 반드시 아래 항목을 체크합니다

```
☑ Add Python 3.12 to PATH   ← 이것 반드시 체크!
☑ Install launcher for all users
```

3. **Install Now** 클릭 → 설치 완료까지 기다립니다

### 1-3. 설치 확인

**시작 메뉴 → 검색창에 `cmd` 입력 → 명령 프롬프트 실행**

아래 명령어를 입력하고 엔터를 누릅니다:

```
python --version
```

아래와 같이 버전이 출력되면 성공입니다:
```
Python 3.12.x
```

⚠ `'python'은(는) 내부 또는 외부 명령...` 오류가 뜨면  
파이썬을 다시 설치하고 **PATH 체크** 항목을 확인하세요.

---

## 2절. 가상환경 venv312 구축

💡 **가상환경이란?**  
프로젝트별로 독립된 파이썬 환경을 만드는 것입니다.  
다른 프로젝트와 라이브러리 버전이 충돌하지 않도록 분리합니다.

### 2-1. 프로젝트 폴더 만들기

명령 프롬프트에서 아래 명령어를 순서대로 입력합니다:

```
cd C:\
mkdir gage_project
cd gage_project
```

💡 `cd` = 폴더 이동, `mkdir` = 폴더 만들기

### 2-2. 가상환경 생성

```
python -m venv venv312
```

실행 후 `gage_project` 폴더 안에 `venv312` 폴더가 생성됩니다.

### 2-3. 가상환경 활성화

```
venv312\Scripts\activate
```

활성화 성공 시 프롬프트 앞에 `(venv312)` 가 표시됩니다:

```
(venv312) C:\gage_project>
```

⚠ 앞으로 작업할 때마다 반드시 **가상환경을 먼저 활성화**하세요!

### 2-4. 가상환경 비활성화 (작업 종료 시)

```
deactivate
```

---

## 3절. 필요한 라이브러리 설치

### 3-1. requirements.txt 파일 만들기

메모장을 열고 아래 내용을 입력한 뒤,  
`C:\gage_project\requirements.txt` 로 저장합니다:

```
torch==2.5.1
torchvision==0.20.1
opencv-python==4.13.0.92
numpy==2.0.2
pandas==2.3.3
matplotlib==3.10.9
Pillow==12.2.0
tqdm==4.67.3
scikit-learn==1.8.0
```

### 3-2. PyTorch (GPU 버전) 설치

💡 CUDA가 설치된 NVIDIA GPU가 있다면 아래 명령어로 GPU 버전 설치:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

💡 GPU가 없다면 (CPU 버전):

```
pip install torch torchvision
```

### 3-3. 나머지 라이브러리 설치

```
pip install -r requirements.txt
```

설치 완료까지 5~15분 정도 소요됩니다.

### 3-4. 설치 확인

```
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.is_available())"
```

출력 예시:
```
PyTorch: 2.5.1
GPU: True
```

---

## 4절. 데이터셋 구조 이해

### 4-1. 폴더 구조

```
C:\gage_project\
├── my_gauge\                   ← 내 게이지 이름 (자유롭게 설정)
│   ├── train\                  ← 학습용 이미지 폴더
│   │   ├── 000-KakaoTalk_xxx.jpg
│   │   ├── 075-KakaoTalk_xxx.jpg
│   │   └── 150-KakaoTalk_xxx.jpg
│   └── test\                   ← 테스트용 이미지 폴더
│       ├── 025-KakaoTalk_xxx.jpg
│       └── 100-KakaoTalk_xxx.jpg
├── requirements.txt
├── step05_preprocess_pipeline.py
└── step06_train_resnet18.py
```

### 4-2. 파일명 규칙 (★ 매우 중요!)

이미지 파일명의 **앞 3자리 숫자**가 게이지 라벨링 값입니다.

| 파일명 예시 | 앞 3자리 | 라벨값(mm) |
|------------|---------|-----------|
| `000-photo.jpg` | 000 | 0.00 mm |
| `050-photo.jpg` | 050 | 0.50 mm |
| `100-photo.jpg` | 100 | 1.00 mm |
| `175-photo.jpg` | 175 | 1.75 mm |

💡 **라벨링 방법:**  
다이얼 게이지를 특정 눈금에 맞춘 상태에서 사진을 찍습니다.  
파일명 앞에 해당 눈금값 × 100 을 3자리로 붙입니다.  
예) 0.75mm → `075-`

### 4-3. 촬영 시 주의사항

- 게이지 정면에서 수직으로 촬영 (기울기 최소화)
- 조명이 고른 환경에서 촬영
- 바늘이 목표 눈금에 정확히 있는지 확인
- 흔들림 없이 촬영 (삼각대 권장)
- 최소 50~100장 이상 촬영 권장 (눈금별 균일하게)

---

## 5절. 전처리 파이프라인 실행

### 5-1. 파이썬 파일 복사

`step05_preprocess_pipeline.py` 파일을  
`C:\gage_project\` 폴더에 복사합니다.

### 5-2. 실행 방법 (방법 1 - 자동 탐색)

```
cd C:\gage_project
venv312\Scripts\activate
python step05_preprocess_pipeline.py
```
