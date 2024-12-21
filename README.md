# RTSS.py (Real-Time Scription & Summarization)
## Introduction
![image](https://github.com/user-attachments/assets/abd8d038-db13-49d5-bda2-554d24ff333c)
RTSS는 학습 및 복습을 위한 강의노트 생성 프로그램으로, 마이크를 통해 실시간으로 오디오 입력을 받아 Whisper를 통해 전사, BART를 통해 요약해 강의 속기록/부분요약/전체요약 이 모두 포함된 강의노트를 생성합니다.

## Feature
- 음성 입력 `sounddevice`를 사용하여 실시간으로 오디오 입력을 받아 처리.
- Whisper ASR 모델을 사용해 음성을 텍스트로 변환
- BART 모델을 사용해 변환된 텍스트를 부분 및 전체 요약
- 속기록, 부분요약 및 최종요약본을 마크다운 파일(`lecture_notes.md`)에 저장.

## Usage

### SETTING
- `DEFAULT_WHISPER` : 사용할 위스퍼 모델을 선택(Transformer 모델만 가능) `default=imTak/whisper_large_v3_turbo_korean_Develop`
- `DEFAULT_SUMMARY` : 사용할 BART 모델을 선택(Transformer 모델만 가능) `default=EbanLee/kobart-summary-v3`
- `CHUNK_DURATION` : 끊어서 인식할 음성 시간 단위 `default=10`
- `DEVICE_INDEX` : 음성 입력 기기 번호 `default=1`
- `OUTPUT_FILE` : 강의노트가 저장될 파일의 이름 `default=lecture_notes.md`
- `LOG_FILE` : 로그가 저장될 파일을 이름 `default=rtsslog.log`

### RUNNING
```bash
python rtss.py
```

### COMMANDS
- `paragraph`: 강의노트에 문단 구분을 추가합니다.
- `exit`: 녹음을 종료하고 전체 요약을 생성합니다. `Ctrl+C` 키를 눌러도 똑같이 작동합니다

## Examples
![image](https://github.com/user-attachments/assets/2919a76b-5f45-451b-86a8-754e8c0e597a)
![image](https://github.com/user-attachments/assets/676eebed-9eb5-47cd-82ca-4366820464b6)
