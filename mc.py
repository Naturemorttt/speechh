from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import gradio as gr
import pyaudio
import wave
import threading
import time
import os
import requests

# Конфигурация записи
RECORD_SECONDS = 15
CHUNK = 1024
FORMAT = pyaudio.paInt32
CHANNELS = 1
RATE = 44100
MICROSERVICE_URL = "http://localhost:8000/transcribe/"

# Глобальные переменные
is_recording = False
frames = []
stream = None
p = None

app = FastAPI()

# Глобальная переменная для хранения текста из POST-запроса
post_text = ""

def wav_to_binary(wav_file_path):
    """Конвертирует WAV в бинарный файл"""
    binary_file_path = os.path.splitext(wav_file_path)[0] + '.bin'
    try:
        with open(wav_file_path, 'rb') as wav_file, open(binary_file_path, 'wb') as binary_file:
            binary_file.write(wav_file.read())
        return binary_file_path
    except Exception as e:
        print(f"Ошибка конвертации: {e}")
        return None

def send_to_microservice(binary_file_path):
    """Отправляет бинарный файл на микросервис транскрибации и возвращает результат"""
    try:
        with open(binary_file_path, 'rb') as f:
            files = {'file': (os.path.basename(binary_file_path), f, 'application/octet-stream')}
            response = requests.post(MICROSERVICE_URL, files=files)
        
        if response.status_code == 200:
            # Получаем текст из ответа
            result = response.json()
            return result.get('transcription', 'Текст не найден в ответе')
        else:
            return f"Ошибка сервера: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Ошибка отправки: {str(e)}"

def record_audio(output_filename):
    global is_recording, frames, stream, p, post_text
    
    if not is_recording:
        # Начало записи
        is_recording = True
        frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, 
                       rate=RATE, input=True,
                       frames_per_buffer=CHUNK)
        
        def recording_thread():
            global is_recording, frames
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                if not is_recording:
                    break
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            is_recording = False
        
        threading.Thread(target=recording_thread).start()
        return "Запись...", post_text  # Возвращаем текущий текст из POST-запроса
    else:
        # Остановка записи
        is_recording = False
        time.sleep(0.1)
        
        if stream:
            stream.stop_stream()
            stream.close()
        if p:
            p.terminate()
        
        result_text = post_text  # Используем текст из POST-запроса по умолчанию
        
        if frames:
            # Сохраняем WAV
            with wave.open(output_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            
            # Конвертируем в бинарный
            binary_path = wav_to_binary(output_filename)
            if binary_path:
                # Отправляем на микросервис и получаем результат
                result_text = send_to_microservice(binary_path)
            
            duration = len(frames)/RATE
            status = f"Запись сохранена ({duration:.1f} сек)"
            return status, result_text
        
        return "Запись отменена", post_text

# Создаем Gradio интерфейс
with gr.Blocks() as demo:
    gr.Markdown("## Запись аудио и транскрибация")
    with gr.Row():
        audio_input = gr.Textbox(label="Имя файла (без расширения)", value="recording")
        record_btn = gr.Button("🎤 Начать запись", variant="primary")
    
    status = gr.Textbox(label="Статус", value="Готов к записи", interactive=False)
    transcription = gr.Textbox(label="Результат транскрибации", interactive=False)
    
    # Скрытый input для автоматического добавления .wav
    wav_input = gr.Textbox(value="recording.wav", visible=False)
    
    record_btn.click(
        fn=record_audio,
        inputs=wav_input,
        outputs=[status, transcription]
    )
    
    audio_input.change(
        fn=lambda x: x + ".wav" if x and not x.endswith(".wav") else x,
        inputs=audio_input,
        outputs=wav_input
    )

# Добавляем endpoint для установки текста
@app.post("/api/set-text/")
async def set_text(request: Request):
    global post_text
    try:
        data = await request.json()
        post_text = data.get("text", "")
        return JSONResponse({
            "status": "success", 
            "text": post_text,
            "message": "Текст будет отображен в поле 'Результат транскрибации'"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Добавляем endpoint для получения текста
@app.get("/api/get-text/")
async def get_text():
    global post_text
    return JSONResponse({"text": post_text})

# Монтируем Gradio приложение
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)