import whisper
import os
import time
from pathlib import Path
import soundfile as sf
import sounddevice as sd
import numpy as np
from whisper.utils import get_writer

def convert_wav_to_mp3(wav_path):
    import subprocess
    mp3_path = wav_path.replace('.wav', '.mp3')
    subprocess.run(['ffmpeg', '-i', wav_path, mp3_path, '-y', '-loglevel', 'quiet'])
    return mp3_path

def monitor_and_transcribe(audio_dir="audio_recordings"):
    model = whisper.load_model("turbo")
    processed_files = set()
    
    while True:
        audio_files = set(f for f in os.listdir(audio_dir) 
                         if f.endswith('.wav') or f.endswith('.mp3'))
        
        new_files = audio_files - processed_files
        for audio_file in new_files:
            audio_path = os.path.join(audio_dir, audio_file)
            srt_path = os.path.join(audio_dir, audio_file.rsplit('.', 1)[0] + '.srt')
            
            if not os.path.exists(srt_path):
                print(f"Transcribing {audio_file}...")
                if audio_file.endswith('.wav'):
                    mp3_path = convert_wav_to_mp3(audio_path)
                    result = model.transcribe(mp3_path)
                else:
                    result = model.transcribe(audio_path)
                
                writer = get_writer("srt", audio_dir)
                writer(result, audio_path)
                print(f"Transcription saved to {srt_path}")
                
            processed_files.add(audio_file)
            
        time.sleep(1)

if __name__ == "__main__":
    monitor_and_transcribe()