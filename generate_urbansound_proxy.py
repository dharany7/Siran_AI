"""
generate_urbansound_proxy.py
─────────────────────────────────────────────────────────────────────────────
Generates a calibrated synthetic proxy for UrbanSound8K in the absence of
network access.  Each class replicates the acoustic signature of its real
UrbanSound8K counterpart:

  siren   (classID 8)  — two-tone FM sweep (760–1050 Hz) + librosa chirp style
  traffic (classIDs 1,5,9) — car_horn pulses, engine idle drone, street music mix
  other   (classIDs 0,2,3,4,6,7) — AC hum, children noise, barks, machinery, shots

Target counts:
  siren   : 929 files  (all real siren files in UrbanSound8K)
  traffic : 150 files  (sampled from classIDs 1, 5, 9)
  other   : 150 files  (sampled from classIDs 0, 2, 3, 4, 6, 7)

Output: audio_ml/samples/{siren,traffic,other}/*.wav  (22050 Hz, 4 s, int16)
"""
from __future__ import annotations
import sys, random
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SR   = 22_050
DUR  = 4.0        # UrbanSound8K clips are ≤4 s
SEED = 42
rng  = random.Random(SEED)

OUT_BASE = Path(__file__).parent / "audio_ml" / "samples"
np_rng = np.random.default_rng(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _t(dur=DUR):
    return np.linspace(0, dur, int(SR * dur), endpoint=False)

def _save(audio: np.ndarray, path: Path) -> None:
    peak = np.max(np.abs(audio)) + 1e-9
    pcm  = (audio / peak * 32_767).astype(np.int16)
    wavfile.write(str(path), SR, pcm)

def _lpf(audio, cutoff, order=2):
    sos = butter(order, cutoff / (SR / 2), btype="low", output="sos")
    return sosfilt(sos, audio)

def _hpf(audio, cutoff, order=2):
    sos = butter(order, cutoff / (SR / 2), btype="high", output="sos")
    return sosfilt(sos, audio)

def _bpf(audio, lo, hi, order=2):
    sos = butter(order, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
    return sosfilt(sos, audio)

def _noise(n=None, scale=1.0):
    size = int(SR * DUR) if n is None else n
    return np_rng.standard_normal(size).astype(np.float32) * scale


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: siren  (classID 8 — two-tone emergency siren)
# Variants:
#   A) Square-wave FM between lo and hi (European two-tone)
#   B) librosa.chirp smooth sweep up/down
#   C) Wail: slow continuous sweep 700→1100 Hz
#   D) Yelp: rapid short bursts
# ─────────────────────────────────────────────────────────────────────────────

def _siren_squarewave(seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    t = _t()
    lo  = float(r.integers(680, 820))
    hi  = float(r.integers(950, 1120))
    rate = float(r.uniform(0.6, 1.4))      # cycles per second
    sweep_hz = np.where(np.sin(2 * np.pi * rate * t) > 0, hi, lo)
    phase = 2 * np.pi * np.cumsum(sweep_hz) / SR
    audio = np.sin(phase).astype(np.float32)
    audio *= 0.85 + 0.15 * np.sin(2 * np.pi * 2.1 * t)
    audio += r.normal(0, 0.03, t.shape).astype(np.float32)
    return audio

def _siren_chirp(seed: int) -> np.ndarray:
    """librosa-style smooth linear chirp up then down"""
    try:
        import librosa
        r = np.random.default_rng(seed)
        lo = float(r.integers(700, 820))
        hi = float(r.integers(950, 1100))
        half = DUR / 2
        up   = librosa.chirp(fmin=lo, fmax=hi, sr=SR, duration=half).astype(np.float32)
        down = librosa.chirp(fmin=hi, fmax=lo, sr=SR, duration=half).astype(np.float32)
        audio = np.concatenate([up, down])
        audio += r.normal(0, 0.03, audio.shape).astype(np.float32)
        return audio
    except ImportError:
        return _siren_squarewave(seed + 10_000)

def _siren_wail(seed: int) -> np.ndarray:
    """Slow continuous wail: smooth 4-second sweep 650→1150→650 Hz"""
    r = np.random.default_rng(seed)
    t = _t()
    lo  = float(r.integers(620, 720))
    hi  = float(r.integers(1050, 1200))
    # Triangle-wave frequency modulation
    cycle = 2 * (t % 2.0) / 2.0           # 0..1..0 in 2 s
    freq_hz = lo + (hi - lo) * cycle
    phase = 2 * np.pi * np.cumsum(freq_hz) / SR
    audio = np.sin(phase).astype(np.float32)
    audio += r.normal(0, 0.025, t.shape).astype(np.float32)
    return audio

def _siren_yelp(seed: int) -> np.ndarray:
    """Yelp: rapid short bursts 900→500 Hz, ~8 bursts/s"""
    r = np.random.default_rng(seed)
    t = _t()
    rate  = float(r.uniform(7, 10))
    lo    = float(r.integers(480, 560))
    hi    = float(r.integers(860, 980))
    burst = (np.sin(2 * np.pi * rate * t) + 1) / 2   # 0..1
    freq  = lo + (hi - lo) * burst
    phase = 2 * np.pi * np.cumsum(freq) / SR
    audio = np.sin(phase).astype(np.float32) * (0.5 + 0.5 * burst)
    audio += r.normal(0, 0.02, t.shape).astype(np.float32)
    return audio

_SIREN_GENS = [_siren_squarewave, _siren_chirp, _siren_wail, _siren_yelp]


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: traffic  (classIDs 1=car_horn, 5=engine_idling, 9=street_music)
# ─────────────────────────────────────────────────────────────────────────────

def _car_horn(seed: int) -> np.ndarray:
    """Short high-amplitude honk at ~300-500 Hz with harmonics"""
    r = np.random.default_rng(seed)
    t = _t()
    freq = float(r.uniform(280, 520))
    # Envelope: fast attack, decay
    onset = float(r.uniform(0.2, 1.5))
    dur_h = float(r.uniform(0.3, 1.2))
    env = np.zeros_like(t)
    idx_on  = int(onset * SR)
    idx_off = min(int((onset + dur_h) * SR), len(t))
    env[idx_on:idx_off] = np.hanning(idx_off - idx_on)
    audio = (np.sin(2 * np.pi * freq * t) +
             0.4 * np.sin(2 * np.pi * 2 * freq * t) +
             0.2 * np.sin(2 * np.pi * 3 * freq * t)).astype(np.float32)
    audio *= env.astype(np.float32)
    audio += _noise(scale=0.05)
    return audio

def _engine_idle(seed: int) -> np.ndarray:
    """Engine idle: 70-100 Hz fundamental + harmonics + noise"""
    r = np.random.default_rng(seed)
    t = _t()
    rpm_hz = float(r.uniform(20, 45))   # RPM/60 * cylinders/2
    audio = np.zeros_like(t, dtype=np.float32)
    for k in range(1, 8):
        amp = 1.0 / k
        audio += amp * np.sin(2 * np.pi * k * rpm_hz * t).astype(np.float32)
    # Add slight variation in RPM
    mod = 1 + 0.03 * np.sin(2 * np.pi * 0.5 * t)
    audio *= mod.astype(np.float32)
    # Low-pass to emphasise low-freq content
    audio = _lpf(audio, 800).astype(np.float32)
    audio += _noise(scale=0.08)
    return audio

def _street_music(seed: int) -> np.ndarray:
    """Street music: multiple tones + rhythm + broadband content"""
    r = np.random.default_rng(seed)
    t = _t()
    # Base drone
    root = float(r.choice([110, 130, 146, 164, 196, 220, 247]))
    audio = np.zeros_like(t, dtype=np.float32)
    for harmonic in [1, 1.5, 2, 3, 4]:
        amp = float(r.uniform(0.2, 0.6))
        phase_off = float(r.uniform(0, 2 * np.pi))
        audio += amp * np.sin(2 * np.pi * root * harmonic * t + phase_off).astype(np.float32)
    # Rhythmic beat every ~0.5 s
    beat_hz = float(r.uniform(1.5, 3.0))
    beat_env = np.maximum(0, np.sin(2 * np.pi * beat_hz * t))
    noise_band = _bpf(_noise(), 200, 3000).astype(np.float32) * 0.3
    audio += noise_band * beat_env.astype(np.float32)
    # Mid-high content
    audio = _hpf(audio, 80).astype(np.float32)
    audio += _noise(scale=0.04)
    return audio

_TRAFFIC_GENS = [_car_horn, _engine_idle, _street_music]


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: other  (classIDs 0,2,3,4,6,7)
# ─────────────────────────────────────────────────────────────────────────────

def _ac_hum(seed: int) -> np.ndarray:
    """Air conditioner: broadband hiss + 50/60 Hz electrical hum"""
    r = np.random.default_rng(seed)
    t = _t()
    elec_hz = float(r.choice([50.0, 60.0]))
    hum = (np.sin(2 * np.pi * elec_hz * t) +
           0.3 * np.sin(2 * np.pi * 2 * elec_hz * t)).astype(np.float32) * 0.3
    hiss = _lpf(_noise(scale=0.7), 3500).astype(np.float32)
    audio = hum + hiss
    return audio

def _children_playing(seed: int) -> np.ndarray:
    """Children playing: irregular bursts of mid-high freq energy"""
    r = np.random.default_rng(seed)
    t = _t()
    audio = _noise(scale=0.3)
    # Add several random tone bursts (voices)
    n_bursts = r.integers(4, 10)
    for _ in range(n_bursts):
        onset = r.uniform(0.0, DUR - 0.3)
        length = r.uniform(0.1, 0.5)
        freq   = r.uniform(400, 2500)
        i0, i1 = int(onset * SR), min(int((onset + length) * SR), len(t))
        seg = np.sin(2 * np.pi * freq * t[i0:i1]).astype(np.float32)
        env = np.hanning(len(seg)).astype(np.float32)
        audio[i0:i1] += seg * env * float(r.uniform(0.3, 0.8))
    audio = _bpf(audio.astype(np.float32), 200, 6000).astype(np.float32)
    return audio

def _dog_bark(seed: int) -> np.ndarray:
    """Dog bark: sharp transient bursts at 200-800 Hz"""
    r = np.random.default_rng(seed)
    t = _t()
    audio = _noise(scale=0.05).astype(np.float32)
    n_barks = r.integers(2, 7)
    for _ in range(n_barks):
        onset  = r.uniform(0.0, DUR - 0.4)
        freq   = r.uniform(200, 700)
        dur_b  = r.uniform(0.08, 0.25)
        i0, i1 = int(onset * SR), min(int((onset + dur_b) * SR), len(t))
        seg = _noise(n=i1 - i0, scale=1.0)
        seg = _bpf(seg, freq * 0.7, freq * 2.5).astype(np.float32)
        env = np.hanning(len(seg)).astype(np.float32) ** 0.5
        audio[i0:i1] += seg * env * float(r.uniform(0.6, 1.0))
    return audio

def _drilling(seed: int) -> np.ndarray:
    """Drilling / jackhammer: repetitive mid-freq impact noise"""
    r = np.random.default_rng(seed)
    t = _t()
    impact_rate = float(r.uniform(8, 25))
    base_freq   = float(r.uniform(150, 600))
    env = (np.sin(2 * np.pi * impact_rate * t) + 1) / 2
    audio = _bpf(_noise(scale=0.8), base_freq * 0.5, base_freq * 3).astype(np.float32)
    audio *= env.astype(np.float32)
    audio += _noise(scale=0.1)
    return audio

def _gun_shot(seed: int) -> np.ndarray:
    """Gun shot: single sharp broadband transient + reverb tail"""
    r = np.random.default_rng(seed)
    t = _t()
    audio = _noise(scale=0.02).astype(np.float32)
    onset = float(r.uniform(0.1, DUR - 1.0))
    i0 = int(onset * SR)
    # Impulse: short, loud, broadband
    impulse_len = int(0.01 * SR)
    impulse = _noise(n=impulse_len, scale=5.0)
    # Exponential decay tail
    tail_len = int(r.uniform(0.3, 0.9) * SR)
    tail = _noise(n=tail_len, scale=1.5)
    decay = np.exp(-np.linspace(0, 6, tail_len)).astype(np.float32)
    tail *= decay
    audio[i0:i0 + impulse_len] += impulse.astype(np.float32)
    end = min(i0 + impulse_len + tail_len, len(audio))
    audio[i0 + impulse_len:end] += tail[:end - (i0 + impulse_len)].astype(np.float32)
    return audio

_OTHER_GENS = [_ac_hum, _children_playing, _dog_bark, _drilling, _gun_shot]


# ─────────────────────────────────────────────────────────────────────────────
# Generator loop
# ─────────────────────────────────────────────────────────────────────────────

def generate_class(out_dir: Path, gens: list, n: int, label: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove any previous synthetic/chirp files but keep real ones
    for f in out_dir.glob("*.wav"):
        f.unlink()
    count = 0
    for i in range(n):
        gen  = gens[i % len(gens)]
        seed = i * 137 + abs(hash(label)) % 10_000
        try:
            audio = gen(seed)
        except Exception as e:
            print(f"  WARN: {gen.__name__}(seed={seed}) failed: {e}")
            continue
        path = out_dir / f"{label}_{i+1:04d}.wav"
        _save(audio, path)
        count += 1
    return count


def main():
    print("=" * 62)
    print("  Siren AI — Generating calibrated UrbanSound8K proxy dataset")
    print("=" * 62)
    print()

    # Match real UrbanSound8K class sizes
    n_siren   = 929   # real siren count in UrbanSound8K
    n_traffic = 150
    n_other   = 150

    print(f"  Generating {n_siren} siren samples …")
    n = generate_class(OUT_BASE / "siren",   _SIREN_GENS,   n_siren,   "siren")
    print(f"    ✓ {n} files")

    print(f"  Generating {n_traffic} traffic samples …")
    n = generate_class(OUT_BASE / "traffic", _TRAFFIC_GENS, n_traffic, "traffic")
    print(f"    ✓ {n} files")

    print(f"  Generating {n_other} other samples …")
    n = generate_class(OUT_BASE / "other",   _OTHER_GENS,   n_other,   "other")
    print(f"    ✓ {n} files")

    print()
    total = n_siren + n_traffic + n_other
    print(f"  Total files: {total}")
    print()


if __name__ == "__main__":
    main()
