"""Physiological preservation metrics used by paper-level reports."""
from __future__ import annotations

import numpy as np


def psd_similarity(reference: np.ndarray, estimate: np.ndarray, sample_rate: float) -> float:
    from scipy.signal import welch
    ref = np.asarray(reference).reshape(-1)
    est = np.asarray(estimate).reshape(-1)
    nperseg = min(256, ref.size, est.size)
    _, ref_psd = welch(ref, fs=sample_rate, nperseg=nperseg)
    _, est_psd = welch(est, fs=sample_rate, nperseg=nperseg)
    ref_psd = ref_psd / max(ref_psd.sum(), 1e-12)
    est_psd = est_psd / max(est_psd.sum(), 1e-12)
    return float(1.0 - 0.5 * np.abs(ref_psd - est_psd).sum())


def band_power(signal: np.ndarray, sample_rate: float) -> dict[str, float]:
    from scipy.signal import welch
    signal = np.asarray(signal).reshape(-1)
    freqs, power = welch(signal, fs=sample_rate, nperseg=min(256, signal.size))
    bands = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
    return {name: float(power[(freqs >= low) & (freqs < high)].sum()) for name, (low, high) in bands.items()}


def band_power_preservation(reference: np.ndarray, estimate: np.ndarray, sample_rate: float) -> dict[str, float]:
    ref, est = band_power(reference, sample_rate), band_power(estimate, sample_rate)
    return {name: float(est[name] / max(ref[name], 1e-12)) for name in ref}
