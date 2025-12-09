import whisper
import os
import time
import subprocess
from pathlib import Path
from whisper.utils import get_writer
import torch
from multiprocessing import Process, Queue

def convert_wav_to_mp3(wav_path):
    """Converts a WAV file to MP3."""
    mp3_path = wav_path.replace('.wav', '.mp3')
    subprocess.run(['ffmpeg', '-i', wav_path, mp3_path, '-y', '-loglevel', 'quiet'])
    return mp3_path

def transcribe_audio(audio_path, model, output_dir):
    """Transcribes an audio file if no existing transcription is found."""
    # Convert first if needed, then compute SRT path based on the actual file we'll transcribe
    if audio_path.endswith('.wav'):
        audio_path = convert_wav_to_mp3(audio_path)
    srt_path = Path(audio_path).with_suffix(".srt")
    out_dir = srt_path.parent

    if srt_path.exists():
        return
    
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(str(audio_path))
    
    writer = get_writer("srt", str(out_dir))
    writer(result, str(audio_path))
    print(f"Transcription saved to {srt_path}")

def _gpu_worker(gpu_id, task_queue):
    """Worker that consumes audio paths and transcribes on a specific device."""
    device = f"cuda:{gpu_id}" if (gpu_id is not None and torch.cuda.is_available()) else "cpu"
    model = whisper.load_model("turbo", device=device)
    while True:
        path = task_queue.get()
        if path is None:
            break
        try:
            transcribe_audio(path, model, os.path.dirname(path))
        except Exception as e:
            print(f"[worker {device}] Error on {path}: {e}")

def _pending_audio_files(audio_dir):
    """Yield audio files lacking an adjacent .srt; prefer mp3 over wav if both exist."""
    for root, _, files in os.walk(audio_dir):
        for fname in files:
            if not fname.endswith(('.wav', '.mp3')):
                continue
            path = os.path.join(root, fname)
            # Skip if an SRT exists for this exact stem
            if Path(path).with_suffix(".srt").exists():
                continue
            # If we see a wav but the mp3 exists, let the mp3 be processed instead
            if fname.endswith('.wav') and Path(path).with_suffix('.mp3').exists():
                continue
            yield path

def monitor_and_transcribe(audio_dir="audio_recordings"):
    """Continuously monitors the directory (recursively) and transcribes any new audio file."""
    # Spin up up to 4 GPU workers (fall back to 1 CPU worker if no GPUs)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    worker_count = min(4, n_gpus) if n_gpus > 0 else 1
    task_q = Queue(maxsize=256)
    workers = []
    for i in range(worker_count):
        p = Process(target=_gpu_worker, args=(i if n_gpus > 0 else None, task_q), daemon=True)
        p.start()
        workers.append(p)

    queued = set()
    try:
        while True:
            # Clean up queued entries that are finished or removed
            for pth in list(queued):
                srt = Path(pth).with_suffix('.srt')
                mp3_srt = Path(Path(pth).with_suffix('.mp3')).with_suffix('.srt')
                if srt.exists() or mp3_srt.exists() or not Path(pth).exists():
                    queued.remove(pth)

            # Enqueue new work
            for path in _pending_audio_files(audio_dir):
                if path not in queued:
                    task_q.put(path)
                    queued.add(path)

            time.sleep(1)  # Prevent excessive CPU usage
    finally:
        # Graceful shutdown
        for _ in workers:
            task_q.put(None)

if __name__ == "__main__":
    # CUDA-safe multiprocessing start method
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    monitor_and_transcribe("study_data")

