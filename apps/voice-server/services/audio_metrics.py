"""
Audio Signal Processing & Objective Quality Assessment Engine.
Computes comprehensive, strictly objective, quantitative metrics across:
1. Amplitude & Waveform (Peak, RMS, Crest Factor, ZCR, DC offset, Clipping, Silence/Speech ratios, Pauses)
2. Spectral & Frequency (FFT, Welch PSD, Centroid, Bandwidth, Rolloff, Flatness, Flux, Dominant Frequencies, Bands)
3. Signal Alignment & Comparative Quality (Cross-correlation lag & Pearson r, SNR, SegSNR, RMSE, SI-SDR, STOI, PESQ)
4. Streaming & Chunking (Boundary discontinuities, Simulated Jitter)
5. Speech Recognition & Domain Accuracy (WER, CER, Medical Terminology hit rate)
6. Diagnostic Multi-Panel Visualization (Waveform, Spectrogram, Energy Envelope, PSD)
"""

import os
import re
import string
import numpy as np
import scipy.signal
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

# Use headless backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pystoi
except ImportError:
    pystoi = None

try:
    from pesq import pesq
except ImportError:
    pesq = None

try:
    import jiwer
except ImportError:
    jiwer = None

EPSILON = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 1. Amplitude & Waveform Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_amplitude_metrics(
    samples: np.ndarray,
    sample_rate: int,
    silence_threshold_dbfs: float = -45.0,
    frame_ms: float = 20.0,
    pause_threshold_ms: float = 200.0
) -> Dict[str, Any]:
    """
    Computes objective waveform and amplitude metrics.
    """
    if len(samples) == 0:
        return {"error": "empty_signal"}

    samples = samples.astype(np.float32)
    abs_samples = np.abs(samples)
    peak_linear = float(np.max(abs_samples))
    peak_dbfs = float(20.0 * np.log10(peak_linear + EPSILON))

    mean_val = float(np.mean(samples))
    std_val = float(np.std(samples))
    rms_linear = float(np.sqrt(np.mean(samples ** 2)))
    rms_dbfs = float(20.0 * np.log10(rms_linear + EPSILON))

    crest_factor = float(peak_linear / (rms_linear + EPSILON))
    crest_factor_db = float(20.0 * np.log10(crest_factor + EPSILON))

    # Zero Crossing Rate (ZCR)
    zero_crossings = np.sum(np.diff(np.signbit(samples)).astype(bool))
    duration_s = len(samples) / sample_rate
    zcr_rate = float(zero_crossings / duration_s) if duration_s > 0 else 0.0

    # Clipping detection (|x| >= 0.999)
    clipping_samples = int(np.sum(abs_samples >= 0.999))
    clipping_ratio = float(clipping_samples / len(samples))

    # Frame-based Silence & Speech analysis
    frame_size = max(1, int(sample_rate * (frame_ms / 1000.0)))
    num_frames = len(samples) // frame_size
    frame_rms_list = []
    silent_frames = 0

    frame_is_silent = []
    for i in range(num_frames):
        chunk = samples[i * frame_size : (i + 1) * frame_size]
        c_rms = np.sqrt(np.mean(chunk ** 2))
        c_dbfs = 20.0 * np.log10(c_rms + EPSILON)
        frame_rms_list.append(c_dbfs)
        is_silent = (c_dbfs < silence_threshold_dbfs)
        frame_is_silent.append(is_silent)
        if is_silent:
            silent_frames += 1

    total_frames = max(1, num_frames)
    silence_ratio = float(silent_frames / total_frames)
    speech_ratio = float(1.0 - silence_ratio)

    # Silence breakdown: Leading, Trailing, Internal pauses
    leading_silent_frames = 0
    for is_sil in frame_is_silent:
        if is_sil:
            leading_silent_frames += 1
        else:
            break

    trailing_silent_frames = 0
    for is_sil in reversed(frame_is_silent):
        if is_sil:
            trailing_silent_frames += 1
        else:
            break

    # Internal pauses (within speech boundaries)
    internal_frames = frame_is_silent[leading_silent_frames : total_frames - trailing_silent_frames]
    internal_silent_frames = sum(internal_frames) if internal_frames else 0

    # Contiguous pause detection (>= pause_threshold_ms)
    min_pause_frames = int((pause_threshold_ms / frame_ms) + 0.5)
    pause_count = 0
    current_run = 0
    for is_sil in internal_frames:
        if is_sil:
            current_run += 1
        else:
            if current_run >= min_pause_frames:
                pause_count += 1
            current_run = 0
    if current_run >= min_pause_frames:
        pause_count += 1

    leading_silence_ms = float(leading_silent_frames * frame_ms)
    trailing_silence_ms = float(trailing_silent_frames * frame_ms)
    internal_silence_ms = float(internal_silent_frames * frame_ms)

    # Dynamic Range (dB): Peak dBFS - Min non-silent frame dBFS
    non_silent_rms = [f for f, s in zip(frame_rms_list, frame_is_silent) if not s]
    if non_silent_rms:
        dynamic_range_db = float(peak_dbfs - min(non_silent_rms))
    else:
        dynamic_range_db = 0.0

    return {
        "duration_seconds": round(duration_s, 4),
        "total_samples": len(samples),
        "peak_amplitude_linear": round(peak_linear, 5),
        "peak_amplitude_dbfs": round(peak_dbfs, 2),
        "rms_amplitude_linear": round(rms_linear, 5),
        "rms_amplitude_dbfs": round(rms_dbfs, 2),
        "crest_factor": round(crest_factor, 3),
        "crest_factor_db": round(crest_factor_db, 2),
        "dc_offset": round(mean_val, 6),
        "std_dev": round(std_val, 5),
        "zero_crossing_rate_hz": round(zcr_rate, 2),
        "clipping_sample_count": clipping_samples,
        "clipping_ratio": round(clipping_ratio, 6),
        "silence_ratio": round(silence_ratio, 4),
        "speech_ratio": round(speech_ratio, 4),
        "leading_silence_ms": round(leading_silence_ms, 1),
        "trailing_silence_ms": round(trailing_silence_ms, 1),
        "internal_silence_ms": round(internal_silence_ms, 1),
        "pause_count_ge_200ms": pause_count,
        "dynamic_range_db": round(dynamic_range_db, 2)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spectral & Frequency Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_spectral_metrics(
    samples: np.ndarray,
    sample_rate: int
) -> Dict[str, Any]:
    """
    Computes spectral distribution and FFT-based frequency metrics using Welch's method.
    """
    if len(samples) < 256:
        return {"error": "signal_too_short_for_spectral"}

    samples = samples.astype(np.float32)
    nperseg = min(1024, len(samples))
    freqs, psd = scipy.signal.welch(samples, fs=sample_rate, nperseg=nperseg, scaling="spectrum")

    total_power = float(np.sum(psd)) + EPSILON

    # Spectral Centroid: sum(f * P(f)) / sum(P(f))
    spectral_centroid = float(np.sum(freqs * psd) / total_power)

    # Spectral Bandwidth: sqrt(sum((f - C)^2 * P(f)) / sum(P(f)))
    spectral_bandwidth = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power))

    # Spectral Rolloff (frequency below which 85% of total power lies)
    cumsum_power = np.cumsum(psd)
    rolloff_idx = np.searchsorted(cumsum_power, 0.85 * total_power)
    rolloff_idx = min(rolloff_idx, len(freqs) - 1)
    spectral_rolloff_85 = float(freqs[rolloff_idx])

    # Spectral Flatness (Wiener entropy): Geometric Mean / Arithmetic Mean of power spectrum
    psd_clamped = np.maximum(psd, EPSILON)
    geom_mean = float(np.exp(np.mean(np.log(psd_clamped))))
    arith_mean = float(np.mean(psd_clamped))
    spectral_flatness = float(geom_mean / (arith_mean + EPSILON))

    # Dominant Frequencies: top 3 peak frequencies in the PSD
    peaks, properties = scipy.signal.find_peaks(psd, distance=5)
    if len(peaks) > 0:
        top_peak_indices = peaks[np.argsort(psd[peaks])[-3:][::-1]]
        dominant_freqs = [round(float(freqs[idx]), 1) for idx in top_peak_indices]
    else:
        dominant_freqs = [round(float(freqs[np.argmax(psd)]), 1)]

    # Band Energy Distribution
    # Band 1: Low band (< 300 Hz)
    # Band 2: Telephone / Core speech (300 Hz - 3400 Hz)
    # Band 3: Wideband speech (3400 Hz - 8000 Hz)
    # Band 4: Ultra-high band (> 8000 Hz)
    low_band_power = float(np.sum(psd[freqs < 300]))
    core_band_power = float(np.sum(psd[(freqs >= 300) & (freqs < 3400)]))
    wide_band_power = float(np.sum(psd[(freqs >= 3400) & (freqs <= 8000)]))
    ultra_band_power = float(np.sum(psd[freqs > 8000]))

    return {
        "spectral_centroid_hz": round(spectral_centroid, 2),
        "spectral_bandwidth_hz": round(spectral_bandwidth, 2),
        "spectral_rolloff_85_hz": round(spectral_rolloff_85, 2),
        "spectral_flatness": round(spectral_flatness, 6),
        "dominant_frequencies_hz": dominant_freqs,
        "band_energy_pct": {
            "low_sub_300hz": round((low_band_power / total_power) * 100.0, 2),
            "core_speech_300_3400hz": round((core_band_power / total_power) * 100.0, 2),
            "wideband_3400_8000hz": round((wide_band_power / total_power) * 100.0, 2),
            "ultra_high_above_8000hz": round((ultra_band_power / total_power) * 100.0, 2)
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Signal Alignment & Comparative Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

def align_signals_cross_correlation(
    ref_signal: np.ndarray,
    deg_signal: np.ndarray,
    sample_rate: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Aligns degraded signal to reference signal via normalized cross-correlation.
    Returns (aligned_ref, aligned_deg, alignment_metadata).
    """
    ref_signal = ref_signal.astype(np.float32)
    deg_signal = deg_signal.astype(np.float32)

    # Demean for correlation
    ref_dm = ref_signal - np.mean(ref_signal)
    deg_dm = deg_signal - np.mean(deg_signal)

    ref_norm = np.linalg.norm(ref_dm)
    deg_norm = np.linalg.norm(deg_dm)

    if ref_norm < EPSILON or deg_norm < EPSILON:
        min_len = min(len(ref_signal), len(deg_signal))
        return ref_signal[:min_len], deg_signal[:min_len], {
            "lag_samples": 0,
            "lag_ms": 0.0,
            "correlation_coefficient": 0.0
        }

    # Cross-correlation: correlate deg against ref
    corr = scipy.signal.correlate(deg_dm, ref_dm, mode="full")
    norm_factor = ref_norm * deg_norm
    corr_normalized = corr / (norm_factor + EPSILON)

    max_idx = int(np.argmax(corr_normalized))
    max_corr = float(corr_normalized[max_idx])

    # Lag in samples: index relative to len(ref) - 1
    lag_samples = max_idx - (len(ref_signal) - 1)
    lag_ms = (lag_samples / sample_rate) * 1000.0

    # Shift and align
    if lag_samples > 0:
        # deg is delayed relative to ref
        aligned_deg = deg_signal[lag_samples:]
        aligned_ref = ref_signal[:len(aligned_deg)]
    elif lag_samples < 0:
        # deg is advanced relative to ref
        aligned_ref = ref_signal[-lag_samples:]
        aligned_deg = deg_signal[:len(aligned_ref)]
    else:
        min_len = min(len(ref_signal), len(deg_signal))
        aligned_ref = ref_signal[:min_len]
        aligned_deg = deg_signal[:min_len]

    # Trim to matching length
    common_len = min(len(aligned_ref), len(aligned_deg))
    aligned_ref = aligned_ref[:common_len]
    aligned_deg = aligned_deg[:common_len]

    return aligned_ref, aligned_deg, {
        "lag_samples": int(lag_samples),
        "lag_ms": round(float(lag_ms), 2),
        "correlation_coefficient": round(float(max_corr), 4)
    }


def calculate_comparative_quality_metrics(
    ref_signal: np.ndarray,
    deg_signal: np.ndarray,
    sample_rate: int
) -> Dict[str, Any]:
    """
    Computes objective signal fidelity metrics: SNR, SegSNR, RMSE, SI-SDR, STOI, PESQ.
    Automatically aligns signals prior to metric calculation.
    """
    ref_aligned, deg_aligned, align_meta = align_signals_cross_correlation(
        ref_signal, deg_signal, sample_rate
    )

    if len(ref_aligned) < 256:
        return {
            "alignment": align_meta,
            "error": "insufficient_aligned_samples"
        }

    # Normalize amplitudes for objective distortion comparison
    ref_energy = np.sum(ref_aligned ** 2) + EPSILON
    deg_energy = np.sum(deg_aligned ** 2) + EPSILON

    # Scale matching factor alpha = <deg, ref> / ||ref||^2
    alpha = float(np.dot(deg_aligned, ref_aligned) / ref_energy)

    # 1. Global SNR (dB)
    error_signal = deg_aligned - ref_aligned
    err_energy = np.sum(error_signal ** 2) + EPSILON
    snr_db = float(10.0 * np.log10(ref_energy / err_energy))

    # 2. Scale-Invariant Signal-to-Distortion Ratio (SI-SDR in dB)
    target_signal = alpha * ref_aligned
    noise_signal = deg_aligned - target_signal
    target_energy = np.sum(target_signal ** 2) + EPSILON
    noise_energy = np.sum(noise_signal ** 2) + EPSILON
    si_sdr_db = float(10.0 * np.log10(target_energy / noise_energy))

    # 3. Root Mean Square Error (RMSE)
    rmse = float(np.sqrt(np.mean(error_signal ** 2)))

    # 4. Segmental SNR (SegSNR) in 20ms frames, clamped to [-10, 35] dB
    frame_len = int(sample_rate * 0.02)
    num_frames = len(ref_aligned) // frame_len
    frame_snrs = []
    for i in range(num_frames):
        r_f = ref_aligned[i * frame_len : (i + 1) * frame_len]
        e_f = error_signal[i * frame_len : (i + 1) * frame_len]
        r_pow = np.sum(r_f ** 2)
        e_pow = np.sum(e_f ** 2) + EPSILON
        if r_pow > 1e-6:  # Only evaluate non-silent frames
            f_snr = 10.0 * np.log10(r_pow / e_pow)
            clamped_snr = max(-10.0, min(35.0, f_snr))
            frame_snrs.append(clamped_snr)
    seg_snr_db = float(np.mean(frame_snrs)) if frame_snrs else snr_db

    # 5. STOI (Short-Time Objective Intelligibility)
    stoi_score = None
    if pystoi is not None:
        try:
            stoi_score = float(pystoi.stoi(ref_aligned, deg_aligned, sample_rate, extended=False))
        except Exception:
            stoi_score = None

    # 6. PESQ (Perceptual Evaluation of Speech Quality - Wideband 16kHz)
    pesq_score = None
    if pesq is not None and sample_rate == 16000:
        try:
            # Scale to [-1.0, 1.0] range for PESQ requirement
            max_val = max(np.max(np.abs(ref_aligned)), np.max(np.abs(deg_aligned)), EPSILON)
            r_norm = (ref_aligned / max_val).astype(np.float32)
            d_norm = (deg_aligned / max_val).astype(np.float32)
            pesq_score = float(pesq(16000, r_norm, d_norm, "wb"))
        except Exception:
            pesq_score = None

    return {
        "alignment": align_meta,
        "snr_db": round(snr_db, 2),
        "segmental_snr_db": round(seg_snr_db, 2),
        "si_sdr_db": round(si_sdr_db, 2),
        "rmse": round(rmse, 6),
        "stoi": round(stoi_score, 4) if stoi_score is not None else None,
        "pesq_wb": round(pesq_score, 3) if pesq_score is not None else None
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Streaming & Chunking Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_chunking_metrics(
    samples: np.ndarray,
    chunk_size_samples: int,
    simulated_latencies_ms: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Evaluates streaming chunk continuity and simulated buffer arrival jitter.
    """
    total_samples = len(samples)
    num_chunks = total_samples // chunk_size_samples

    if num_chunks < 2:
        return {
            "chunk_count": num_chunks,
            "chunk_size_samples": chunk_size_samples,
            "max_boundary_discontinuity": 0.0,
            "mean_boundary_discontinuity": 0.0,
            "inter_chunk_jitter_ms": 0.0
        }

    # Check sample step across chunk seams
    boundary_diffs = []
    for k in range(1, num_chunks):
        boundary_idx = k * chunk_size_samples
        diff = abs(float(samples[boundary_idx]) - float(samples[boundary_idx - 1]))
        boundary_diffs.append(diff)

    max_disc = float(np.max(boundary_diffs)) if boundary_diffs else 0.0
    mean_disc = float(np.mean(boundary_diffs)) if boundary_diffs else 0.0

    # Jitter calculation from chunk arrival times if provided
    jitter_ms = 0.0
    if simulated_latencies_ms and len(simulated_latencies_ms) > 1:
        jitter_ms = float(np.std(simulated_latencies_ms))

    return {
        "chunk_count": num_chunks,
        "chunk_size_samples": chunk_size_samples,
        "max_boundary_discontinuity": round(max_disc, 5),
        "mean_boundary_discontinuity": round(mean_disc, 5),
        "inter_chunk_jitter_ms": round(jitter_ms, 3)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Text Recognition & Medical Terminology Metrics
# ─────────────────────────────────────────────────────────────────────────────

def normalize_clinical_text(text: str) -> str:
    """Normalizes clinical text for objective WER/CER comparison."""
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


CLINICAL_SYNONYMS = {
    "milligrams": ["mg", "milligram", "milligrams"],
    "milligram": ["mg", "milligram", "milligrams"],
    "mg": ["mg", "milligram", "milligrams"],
    "micrograms": ["mcg", "microgram", "micrograms"],
    "microgram": ["mcg", "microgram", "micrograms"],
    "mcg": ["mcg", "microgram", "micrograms"],
    "milliliters": ["ml", "milliliter", "milliliters"],
    "milliliter": ["ml", "milliliter", "milliliters"],
    "ml": ["ml", "milliliter", "milliliters"],
    "kilograms": ["kg", "kilogram", "kilograms"],
    "ecg": ["ecg", "ekg", "electrocardiogram"],
    "ekg": ["ecg", "ekg", "electrocardiogram"],
    "bid": ["bid", "twice daily", "twice a day"],
    "tid": ["tid", "three times daily", "three times a day"],
    "prn": ["prn", "as needed"],
    "otc": ["otc", "over the counter"],
    "bp": ["bp", "blood pressure"],
    "spo2": ["spo2", "sp o2", "o2 saturation", "oxygen saturation"]
}


def calculate_recognition_metrics(
    reference_text: str,
    hypothesis_text: str,
    target_medical_terms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes WER, CER, and domain-specific medical terminology recognition accuracy.
    """
    ref_norm = normalize_clinical_text(reference_text)
    hyp_norm = normalize_clinical_text(hypothesis_text)

    wer = float(jiwer.wer(ref_norm, hyp_norm)) if jiwer else 0.0
    cer = float(jiwer.cer(ref_norm, hyp_norm)) if jiwer else 0.0

    # Medical terminology evaluation
    med_results = {}
    if target_medical_terms:
        hyp_words = set(hyp_norm.split())
        matched_terms = []
        missed_terms = []

        for term in target_medical_terms:
            norm_term = normalize_clinical_text(term)
            variants = CLINICAL_SYNONYMS.get(norm_term, [norm_term])
            
            # Check if any variant is present in hypothesis
            term_found = False
            for v in variants:
                if " " in v:
                    if v in hyp_norm:
                        term_found = True
                        break
                else:
                    if v in hyp_words:
                        term_found = True
                        break
            
            if term_found:
                matched_terms.append(term)
            else:
                missed_terms.append(term)

        total_terms = len(target_medical_terms)
        term_accuracy = len(matched_terms) / total_terms if total_terms > 0 else 1.0
        med_results = {
            "total_target_terms": total_terms,
            "matched_terms_count": len(matched_terms),
            "matched_terms": matched_terms,
            "missed_terms_count": len(missed_terms),
            "missed_terms": missed_terms,
            "medical_term_accuracy": round(term_accuracy, 4)
        }

    return {
        "reference_normalized": ref_norm,
        "hypothesis_normalized": hyp_norm,
        "word_error_rate": round(wer, 4),
        "character_error_rate": round(cer, 4),
        "medical_terminology": med_results
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Failure & Outlier Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_outliers_and_failures(
    metrics: Dict[str, Any],
    wer_threshold: float = 0.20,
    stoi_threshold: float = 0.75,
    pesq_threshold: float = 2.0,
    clipping_threshold: float = 0.001
) -> List[str]:
    """
    Flags objective pipeline anomalies, failure modes, or quality regressions.
    """
    flags = []

    # WER outlier
    wer = metrics.get("recognition", {}).get("word_error_rate", 0.0)
    if wer > wer_threshold:
        flags.append(f"HIGH_WER (WER={wer:.3f} > {wer_threshold})")

    # Missed Medical Terms
    med_meta = metrics.get("recognition", {}).get("medical_terminology", {})
    missed = med_meta.get("missed_terms", [])
    if missed:
        flags.append(f"MISSED_MEDICAL_TERMS ({', '.join(missed)})")

    # Clipping detection
    clipping_ratio = metrics.get("amplitude", {}).get("clipping_ratio", 0.0)
    if clipping_ratio > clipping_threshold:
        flags.append(f"CLIPPING_DETECTED (ratio={clipping_ratio:.5f})")

    # Quality degradation
    stoi = metrics.get("comparative_quality", {}).get("stoi")
    if stoi is not None and stoi < stoi_threshold:
        flags.append(f"LOW_STOI (STOI={stoi:.3f} < {stoi_threshold})")

    pesq_wb = metrics.get("comparative_quality", {}).get("pesq_wb")
    if pesq_wb is not None and pesq_wb < pesq_threshold:
        flags.append(f"LOW_PESQ (PESQ={pesq_wb:.2f} < {pesq_threshold})")

    # Silence anomalies
    silence_ratio = metrics.get("amplitude", {}).get("silence_ratio", 0.0)
    if silence_ratio > 0.60:
        flags.append(f"EXCESSIVE_SILENCE (silence_ratio={silence_ratio:.2f})")

    # VAD Premature Cutoff
    if metrics.get("vad", {}).get("premature_cutoff_detected", False):
        flags.append("VAD_PREMATURE_CUTOFF")

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# 7. Diagnostic Multi-Panel Plot Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_diagnostic_plot(
    ref_samples: np.ndarray,
    deg_samples: np.ndarray,
    sample_rate: int,
    output_png_path: str,
    test_id: str,
    title: str = "Audio Pipeline Diagnostic"
) -> str:
    """
    Generates an objective 4-panel diagnostic plot:
    Panel 1: Aligned Waveforms (Reference vs Pipeline Processed)
    Panel 2: Spectrogram (0 - 8 kHz)
    Panel 3: Short-Time RMS Energy Envelope with Thresholds
    Panel 4: Power Spectral Density (Welch PSD) with Speech Band Markers
    """
    Path(output_png_path).parent.mkdir(parents=True, exist_ok=True)

    ref_aligned, deg_aligned, _ = align_signals_cross_correlation(
        ref_samples, deg_samples, sample_rate
    )
    time_axis = np.linspace(0, len(ref_aligned) / sample_rate, len(ref_aligned))

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f"{title} - Test ID: {test_id}", fontsize=14, fontweight="bold")

    # ── Panel 1: Waveform Comparison ──
    ax1 = axes[0]
    ax1.plot(time_axis, ref_aligned, label="Reference (TTS Resampled 16k)", color="#1f77b4", alpha=0.7, linewidth=0.8)
    ax1.plot(time_axis, deg_aligned, label="Processed (Pipeline / STT In)", color="#ff7f0e", alpha=0.7, linewidth=0.8, linestyle="--")
    ax1.axhline(0.99, color="red", linestyle=":", alpha=0.6, label="Clipping (+/-0.99)")
    ax1.axhline(-0.99, color="red", linestyle=":", alpha=0.6)
    ax1.set_title("1. Aligned Time-Domain Waveforms", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8)

    # ── Panel 2: Spectrogram (Processed Audio) ──
    ax2 = axes[1]
    Pxx, freqs, bins, im = ax2.specgram(deg_aligned, NFFT=512, Fs=sample_rate, noverlap=256, cmap="magma")
    ax2.set_title("2. Spectrogram (0 - 8 kHz)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_ylim(0, sample_rate // 2)
    fig.colorbar(im, ax=ax2, format="%+2.0f dB", pad=0.01)

    # ── Panel 3: Short-Time RMS Energy Envelope ──
    ax3 = axes[2]
    frame_len = int(sample_rate * 0.02)  # 20ms
    num_frames = len(deg_aligned) // frame_len
    t_frames = np.linspace(0, len(deg_aligned) / sample_rate, num_frames)
    rms_frames = []
    for i in range(num_frames):
        ch = deg_aligned[i * frame_len : (i + 1) * frame_len]
        r = np.sqrt(np.mean(ch ** 2))
        rms_frames.append(20.0 * np.log10(r + EPSILON))
    rms_frames = np.array(rms_frames)

    ax3.plot(t_frames, rms_frames, color="#2ca02c", linewidth=1.2, label="Processed Frame RMS (20ms)")
    ax3.axhline(-45.0, color="gray", linestyle="--", label="Silence Threshold (-45 dBFS)")
    ax3.set_title("3. Short-Time RMS Energy Envelope", fontsize=11, fontweight="semibold")
    ax3.set_ylabel("Energy (dBFS)")
    ax3.set_ylim(-80, 5)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=8)

    # ── Panel 4: Power Spectral Density (Welch PSD) ──
    ax4 = axes[3]
    f_ref, p_ref = scipy.signal.welch(ref_aligned, fs=sample_rate, nperseg=min(1024, len(ref_aligned)))
    f_deg, p_deg = scipy.signal.welch(deg_aligned, fs=sample_rate, nperseg=min(1024, len(deg_aligned)))

    ax4.semilogy(f_ref, p_ref + EPSILON, label="Reference PSD", color="#1f77b4", linewidth=1.0)
    ax4.semilogy(f_deg, p_deg + EPSILON, label="Processed PSD", color="#ff7f0e", linestyle="--", linewidth=1.0)
    ax4.axvspan(300, 3400, color="green", alpha=0.08, label="Core Speech Band (300-3400 Hz)")
    ax4.axvspan(3400, 8000, color="purple", alpha=0.05, label="Wideband Band (3400-8000 Hz)")
    ax4.set_title("4. Power Spectral Density (PSD)", fontsize=11, fontweight="semibold")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Power / Freq (V^2/Hz)")
    ax4.set_xlim(0, sample_rate // 2)
    ax4.grid(True, which="both", linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", fontsize=8)

    plt.savefig(output_png_path, dpi=120)
    plt.close(fig)
    return str(output_png_path)
