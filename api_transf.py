from fastapi import FastAPI, UploadFile, File, HTTPException
import whisper
import torch
import os
from typing import Optional
import json
import wave
import numpy as np
import librosa
import time
import ffmpeg
import pyaudio
from pydub import AudioSegment
import warnings
import requests

warnings.filterwarnings('ignore')

print('Импорт ФФФФ')
ffmpeg_path = f"ffmpeg"
os.environ["PATH"] += os.pathsep + ffmpeg_path

print('Импорт модели')

from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Загружаем модель и процессор
model_name = "openai/whisper-large"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

print('Модель импортирована')

print('Запуск апихи')
# Перемещаем модель на GPU, если доступен
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

app = FastAPI()


def binary_to_wav(binary_data: bytes, wav_file_path: str) -> Optional[str]:
    try:
        # Записываем бинарные данные в .wav файл
        with open(wav_file_path, 'wb') as wav_file:
            wav_file.write(binary_data)
        return wav_file_path
    except Exception as e:
        print(f"Произошла ошибка при конвертации: {e}")
        return None

def transcribe_audio(file_path: str) -> Optional[str]:
    try:
        # Проверяем расширение файла
        if file_path.endswith('.bin'):
            wav_file = file_path.replace('.bin', '.wav')
            with open(file_path, 'rb') as f:
                binary_data = f.read()
            file_path = binary_to_wav(binary_data, wav_file)
            if not file_path:
                return None
            
        audio, sr = librosa.load(file_path, sr=16000)  # Whisper требует 16кГц
        input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)

        # Создаем forced_decoder_ids для русского языка
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="russian", task="transcribe")

        # Генерируем токены
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids
        )

        # Декодируем в текст
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
        return transcription
    
    except Exception as e:
        print(f"Ошибка при транскрибации: {e}")
        return None
    finally:
        # Удаляем временные файлы
        if file_path.endswith('.wav') and os.path.exists(file_path):
            os.remove(file_path)

@app.post("/transcribe/")
async def transcribe_binary(file: UploadFile = File(...)):
    if not file.filename.endswith('.bin'):
        raise HTTPException(status_code=400, detail="Требуется файл с расширением .bin")
    
    try:
        # Сохраняем бинарный файл временно
        temp_bin_path = f"temp_{file.filename}"
        with open(temp_bin_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Транскрибируем аудио
        transcription = transcribe_audio(temp_bin_path)
        
        if transcription is None:
            raise HTTPException(status_code=500, detail="Ошибка при обработке аудио")
        elif transcription == "Субтитры сделал DimaTorzok":
            transcription = "Ошибка, низкое качество голосового сообщения." 
        elif transcription == " Продолжение следует...":
            transcription = "Ошибка, низкое качество голосового сообщения." 
        
        # в (8002)
        try:
            response = requests.post(
                "http://localhost:8002/api/set-text/",  
                json={"text": transcription}
            )
            response.raise_for_status()  
        except requests.RequestException as e:
            print(f"Ошибка при отправке: {e}")
        

        return {"transcription": transcription}
    
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_bin_path):
            os.remove(temp_bin_path)

if __name__ == "__main__":
    import uvicorn
    from uvicorn.config import Config
    
    config = Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    
  
    import nest_asyncio
    nest_asyncio.apply()
    server.run()