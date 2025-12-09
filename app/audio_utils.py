import io
import subprocess
from typing import Union


def convert_webm_to_wav(webm_data: Union[bytes, bytearray]) -> io.BytesIO:
    """
    Convert WebM audio bytes to a WAV byte stream that Whisper expects.

    - Resamples to 16kHz, mono
    - Uses ffmpeg via stdin/stdout to avoid temp files
    """
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        "pipe:1",
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wav_data, stderr = process.communicate(input=bytes(webm_data))

    # Raise if ffmpeg fails or produces empty output (common when wrong mime/empty buffer)
    if process.returncode != 0 or not wav_data:
        err = (stderr or b"").decode("utf-8", "ignore")
        raise ValueError(f"ffmpeg failed: code={process.returncode}, stderr={err[:200]!r}")

    return io.BytesIO(wav_data)


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> io.BytesIO:
    """
    Build a WAV BytesIO from raw PCM bytes.
    """
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    buf.seek(0)
    return buf
