import whisper
import os
import time
import subprocess
from pathlib import Path
from whisper.utils import get_writer

def convert_wav_to_mp3(wav_path):
    """Converts a WAV file to MP3."""
    mp3_path = wav_path.replace('.wav', '.mp3')
    subprocess.run(['ffmpeg', '-i', wav_path, mp3_path, '-y', '-loglevel', 'quiet'])
    return mp3_path

def transcribe_audio(audio_path, model, output_dir):
    """Transcribes an audio file if no existing transcription is found."""
    srt_path = Path(output_dir) / (Path(audio_path).stem + ".srt")
    
    if srt_path.exists():
        #print(f"Skipping {audio_path}, transcription already exists.")
        return
    
    if audio_path.endswith('.wav'):
        audio_path = convert_wav_to_mp3(audio_path)
    
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(str(audio_path))
    
    writer = get_writer("srt", output_dir)
    writer(result, str(audio_path))
    print(f"Transcription saved to {srt_path}")

def monitor_and_transcribe(audio_dir="audio_recordings"):
    """Continuously monitors the directory and transcribes any new audio file."""
    model = whisper.load_model("turbo")
    
    while True:
        for audio_file in os.listdir(audio_dir):
            if audio_file.endswith(('.wav', '.mp3')):
                audio_path = os.path.join(audio_dir, audio_file)
                transcribe_audio(audio_path, model, audio_dir)
        
        time.sleep(1)  # Prevent excessive CPU usage

if __name__ == "__main__":
    monitor_and_transcribe()

