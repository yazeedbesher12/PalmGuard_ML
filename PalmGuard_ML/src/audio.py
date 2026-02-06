from __future__ import annotations

from math import gcd
from typing import List, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, stft


def wav_load(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load WAV as mono float32 in [-1, 1], resampling to target_sr if needed."""
    sr, x = wavfile.read(path)

    # Convert to float32 [-1, 1]
    if x.dtype.kind in ("i", "u"):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float32)

    # Stereo -> mono
    if x.ndim == 2:
        x = x.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        up = int(target_sr // g)
        down = int(sr // g)
        x = resample_poly(x, up, down).astype(np.float32)
        sr = target_sr

    return x, sr


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float, fmax: float) -> np.ndarray:
    """Create mel filterbank matrix (n_mels, n_fft//2 + 1)."""
    n_freqs = n_fft // 2 + 1

    m_min = float(hz_to_mel(np.array([fmin], dtype=np.float32))[0])
    m_max = float(hz_to_mel(np.array([fmax], dtype=np.float32))[0])
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float32)
    f_pts = mel_to_hz(m_pts)

    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])

        if center <= left:
            center = min(left + 1, n_freqs - 1)
        if right <= center:
            right = min(center + 1, n_freqs - 1)

        if center - left > 0:
            fb[i, left:center] = (np.arange(left, center) - left) / (center - left)
        if right - center > 0:
            fb[i, center:right] = (right - np.arange(center, right)) / (right - center)

    return fb


def logmel_from_waveform(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
    n_mels: int = 64,
    fmin: float = 50.0,
    fmax: float = 3500.0,
) -> np.ndarray:
    """Compute log-mel spectrogram (n_mels, T)."""
    noverlap = win_length - hop_length
    _, _, Zxx = stft(
        y,
        fs=sr,
        nperseg=win_length,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    P = (np.abs(Zxx) ** 2).astype(np.float32)  # (n_freqs, T)

    fb = mel_filterbank(n_mels=n_mels, n_fft=n_fft, sr=sr, fmin=fmin, fmax=fmax)
    M = (fb @ P).astype(np.float32)  # (n_mels, T)

    return np.log10(M + 1e-10).astype(np.float32)


def logmel_segments(
    y: np.ndarray,
    sr: int,
    *,
    seg_s: float = 2.0,
    hop_s: float = 1.0,
    n_mels: int = 64,
    fmin: float = 50.0,
    fmax: float = 3500.0,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
    normalize: bool = True,
) -> List[np.ndarray]:
    """Split waveform into segments and return list of (1, n_mels, T) log-mel tensors."""
    seg_len = int(seg_s * sr)
    hop_len = int(hop_s * sr)
    if seg_len <= 0:
        raise ValueError("seg_s must be > 0")

    max_start = max(0, len(y) - seg_len)
    starts = list(range(0, max_start + 1, hop_len)) if max_start > 0 else [0]

    out: List[np.ndarray] = []
    for start in starts:
        seg = y[start:start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)), mode="constant")

        logM = logmel_from_waveform(
            seg,
            sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        if normalize:
            logM = (logM - float(logM.mean())) / (float(logM.std()) + 1e-6)

        out.append(logM[None, :, :])  # (1, n_mels, T)

    return out
