import sounddevice as sd, numpy as np
import whisper, threading, queue, logging, torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, WhisperProcessor, WhisperForConditionalGeneration, PreTrainedTokenizerFast, BartForConditionalGeneration

# 설정
DEFAULT_WHISPER = "imTak/whisper_large_v3_turbo_korean_Develop"
DEFAULT_SUMMARY = "EbanLee/kobart-summary-v3"
SAMPLE_RATE = 16000
CHUNK_DURATION = 10
DEVICE_INDEX = 1
OUTPUT_FILE = "lecture_notes.md" # 강의노트 출력
LOG_FILE = "rtsslog.log" # 로그파일

# 로거
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 전역
stop_signal = threading.Event()  # 프로그램 종료 신호
audio_queue = queue.Queue()      # 오디오 데이터를 처리하기 위한 큐
command_queue = queue.Queue()    # 사용자 명령을 처리하기 위한 큐
text_buffer = []                 # 누적된 텍스트 저장 리스트

# --------------------
# 모델 로드
# --------------------
def load_whisper_model(model_name=DEFAULT_WHISPER):
    logging.info(f"Whisper 모델 로드 중: {model_name}")
    processor, model = (AutoProcessor.from_pretrained(model_name),
                        AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Whisper 모델 로드 완료")
    return processor, model

def load_summary_model(model_name=DEFAULT_SUMMARY):
    logging.info(f"요약 모델 로드 중: {model_name}")
    tokenizer, model = (PreTrainedTokenizerFast.from_pretrained(model_name),
                        BartForConditionalGeneration.from_pretrained(model_name))
    logging.info("요약 모델 로드 완료")
    return tokenizer, model

# --------------------
# 요약
# --------------------
def summarize_text_kobart(text_buffer, tokenizer, model, chunk_size=600):
    """
    KoBART를 사용하여 부분 요약 및 최종 요약을 생성합니다.
    """
    def _generate_summary(text):
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=chunk_size)
        summary_ids = model.generate(
            input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
            max_length=150, min_length=12, num_beams=6, no_repeat_ngram_size=15,
            repetition_penalty=1.5
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    summaries, chunk = [], ""
    for text in text_buffer:
        chunk = f"{chunk} {text}".strip() if len(chunk) + len(text) <= chunk_size else summaries.append(_generate_summary(chunk)) or text
    if chunk: summaries.append(_generate_summary(chunk))
    return summaries, _generate_summary(" ".join(summaries))

# --------------------
# 강의노트 저장
# --------------------
def write_to_markdown(text, paragraph=False):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        if paragraph:
            f.write("\n\n")  # 문단 구분
        f.write(f"{text.strip()} ")

def save_summary(text_buffer):
    tokenizer, model = load_summary_model()
    summaries, final_summary = summarize_text_kobart(text_buffer, tokenizer, model)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\n# 부분 요약\n")
        f.writelines([f"\n- {summary}" for i, summary in enumerate(summaries)])
        f.write(f"\n\n# 최종 요약\n\n{final_summary}\n\n")

# --------------------
# 오디오 입력
# --------------------
def audio_callback(indata, frames, time, status):
    if status:
        logging.warning(f"Audio Callback Status: {status}")
    try:
        audio_queue.put(indata.copy())
        logging.debug(f"Audio Callback: Received {len(indata)} frames.")
    except Exception as e:
        logging.error(f"Audio Callback Exception: {e}")

def audio_input_thread(device_index):
    logging.info("오디오 입력 스레드 시작")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
            device=device_index, channels=1, 
            callback=audio_callback,
        ):
            logging.info("오디오 입력 스트림 시작됨")
            while not stop_signal.is_set():
                stop_signal.wait(1)
    except Exception as e:
        logging.error(f"오디오 입력 스레드 오류: {e}")

# --------------------
# 오디오 Transcribe
# --------------------
def transcribe_audio(audio_data, processor, model, language="ko"):
    try:
        logging.debug("오디오를 텍스트로 변환 중...")
        input_features = processor(audio_data, return_tensors="pt", sampling_rate=SAMPLE_RATE).input_features.cuda()
        predicted_ids = model.generate(input_features=input_features, num_beams=5, repetition_penalty=1.5)
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    except Exception as e:
        logging.error(f"음성 변환 중 오류 발생: {e}")
        return "변환 실패"

def audio_processing_thread():
    processor, model = load_whisper_model()
    audio_buffer = []
    while not stop_signal.is_set():
        try:
            audio_chunk = audio_queue.get(timeout=1)
            audio_buffer.append(audio_chunk)

            if len(audio_buffer) * len(audio_chunk) >= SAMPLE_RATE * CHUNK_DURATION:
                logging.debug("오디오 데이터 변환 중...")
                audio_data = np.concatenate(audio_buffer, axis=0).flatten()
                audio_buffer = []  # 버퍼 초기화

                # Whisper 변환
                result = transcribe_audio(audio_data, processor, model, language="ko")
                logging.info(f"변환된 텍스트: {result}")
                text_buffer.append(result)
                write_to_markdown(result)  # 텍스트 저장
        except queue.Empty:
            logging.debug("audio_queue가 비어 있습니다. 데이터를 기다리는 중...")
        except Exception as e:
            logging.error(f"오디오 처리 스레드 오류: {e}")

# --------------------
# 유저 명령
# --------------------
def command_thread():
    while not stop_signal.is_set():
        try:
            command = input("").strip().lower()
            if command == "paragraph":
                command_queue.put("paragraph")
            elif command == "exit":
                logging.info("종료 명령어가 입력되었습니다.")
                stop_signal.set()
            else:
                logging.warning(f"알 수 없는 명령어: {command}")
        except Exception as e:
            logging.error(f"사용자 명령어 처리 중 오류 발생: {e}")

# --------------------
# 메인
# --------------------
def main():
    logging.info("프로그램 시작")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# 실시간 강의 노트\n\n")

    for t in (threads := [
        threading.Thread(target=audio_input_thread, args=(DEVICE_INDEX,), daemon=True),
        threading.Thread(target=audio_processing_thread, daemon=True),
        threading.Thread(target=command_thread, daemon=True),
    ]):t.start() 

    try:
        while not stop_signal.is_set():
            # 사용자 명령 처리
            while not command_queue.empty():
                command = command_queue.get()
                if command == "paragraph":
                    logging.info("문단 분리 명령어 실행")
                    write_to_markdown("", paragraph=True)
            stop_signal.wait(1)
    except KeyboardInterrupt:
        logging.info("Ctrl+C 감지. 프로그램 종료 중...")
        stop_signal.set()
    finally:
        [t.join() for t in threads]

        # 전체 텍스트 요약 저장
        if text_buffer:
            save_summary(text_buffer)

        logging.info("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()