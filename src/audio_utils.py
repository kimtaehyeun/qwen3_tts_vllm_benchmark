import io
import os
import wave
from datetime import datetime
from typing import Optional

import librosa
import soundfile as sf


def ensure_directory(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_audio_bytes(output_path: str, audio_bytes: bytes) -> None:
    ensure_directory(output_path)
    with open(output_path, 'wb') as output_file:
        output_file.write(audio_bytes)


def get_audio_duration(output_path: str) -> Optional[float]:
    try:
        info = sf.info(output_path)
        if info.frames is not None and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        duration = librosa.get_duration(filename=output_path)
        return float(duration)
    except Exception:
        return None


def get_audio_duration_from_bytes(audio_bytes: bytes) -> Optional[float]:
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if frame_rate:
                return float(frame_count) / float(frame_rate)
    except Exception:
        pass
    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as sound_file:
            if sound_file.frames is not None and sound_file.samplerate:
                return float(sound_file.frames) / float(sound_file.samplerate)
    except Exception:
        pass
    try:
        temp_buffer = io.BytesIO(audio_bytes)
        duration = librosa.get_duration(filename=temp_buffer)
        return float(duration)
    except Exception:
        return None
