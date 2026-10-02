"""Shared CPU runtime. All writable state stays under this project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
(CACHE / "tmp").mkdir(parents=True, exist_ok=True)  # phonemizer needs TMPDIR to exist
for name, relative in {
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "HF_HOME": str(CACHE / "hf"),
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "AUDIOSEAL_CACHE_DIR": str(CACHE),
    "XDG_CACHE_HOME": str(CACHE),
    "TORCH_HOME": str(CACHE / "torch"),
    "TMPDIR": str(CACHE / "tmp"),
}.items():
    os.environ[name] = relative

import hashlib
import shutil
import json
import time
import importlib.metadata
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort
from scipy.signal import resample_poly
import espeakng_loader
from kokoro_onnx import Kokoro
from kokoro_onnx.config import EspeakConfig
from audioseal import AudioSeal
from faster_whisper import WhisperModel

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.manual_seed(20261002)
SR = 16000
MESSAGE = [int(x) for x in "1010010110100101"]


def espeak_config():
    """espeak-ng fails (prints a bogus phontab error and the process dies) when the espeakng_loader
    package sits under a very long path, as in a deeply nested venv. Observed here with a 194
    character data path; a 159 character one worked. If the path is long, run from a short copy."""
    data, lib = espeakng_loader.get_data_path(), espeakng_loader.get_library_path()
    if len(data) < 150:
        return EspeakConfig(data_path=data, lib_path=lib)
    short = Path(f"/tmp/espeakng_loader-{os.getuid()}")
    if not short.exists():
        shutil.copytree(Path(lib).parent, short)
    return EspeakConfig(data_path=str(short / "espeak-ng-data"), lib_path=str(short / Path(lib).name))


def tts_model():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(CACHE / "tts/kokoro-v1.0.onnx"),
                                  sess_options=options, providers=["CPUExecutionProvider"])
    return Kokoro.from_session(session, str(CACHE / "tts/voices-v1.0.bin"),
                               espeak_config=espeak_config())


def synth(tts, text, voice, speed=1.0):
    audio, sr = tts.create(text, voice=voice, speed=speed,
                           lang="en-gb" if voice.startswith("b") else "en-us")
    assert sr == 24000 and len(audio) and np.max(np.abs(audio)) > 0.01
    audio = resample_poly(audio, 2, 3).astype(np.float32)
    return np.pad(audio, (640, 640))


def dump(path, obj):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def models():
    generator = AudioSeal.load_generator("audioseal_wm_16bits", device="cpu").eval()
    detector = AudioSeal.load_detector("audioseal_detector_16bits", device="cpu").eval()
    asr = WhisperModel("tiny", device="cpu", compute_type="int8",
                       cpu_threads=2, num_workers=1, download_root=str(CACHE / "whisper"))
    return generator, detector, asr


def mark(generator, audio):
    wav = torch.from_numpy(audio.copy())[None, None]
    with torch.inference_mode():
        residual = generator.get_watermark(wav, message=torch.tensor([MESSAGE]))
    return (wav + residual)[0, 0].numpy().copy()


def detect(detector, audio):
    with torch.inference_mode():
        probability, bits = detector(torch.from_numpy(audio.copy())[None, None])
    p = probability[0, 1].numpy().copy()
    return p, (bits[0].numpy() >= 0.5).astype(int).tolist()


def transcribe(asr, audio):
    segments, _ = asr.transcribe(audio, language="en", beam_size=1, temperature=0,
                                word_timestamps=True, vad_filter=False,
                                condition_on_previous_text=False)
    return [{"word": w.word, "start": w.start, "end": w.end}
            for segment in segments for w in segment.words]
