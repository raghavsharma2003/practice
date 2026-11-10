#!/usr/bin/env python3
"""
"Low Orbit" - an original minimal / progressive (melodic) techno track,
synthesised from scratch with numpy + scipy (no samples).

124 BPM, G minor, 192 bars (~6:12).

Outputs (in techno/output/):
  low_orbit_full_mix.mp3      full arranged + mastered track
  low_orbit_full_mix.wav      (not committed - large)
  loops/<stem>_8bar.wav       8-bar stem loops from the main drop, DAW ready
  loops/full_loop_8bar.wav    all stems summed for that 8-bar section

Usage:  python3 techno/generate.py
"""
import os
import subprocess
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d

SR = 44100
BPM = 124
BEAT = 60.0 / BPM
STEP = BEAT / 4          # 16th note
BAR = BEAT * 4
N_BARS = 192
TAIL = 4.0               # seconds of tail after last bar
TOTAL = int((N_BARS * BAR + TAIL) * SR)
SWING = 0.08             # 54% MPC-style swing on hats/perc (fraction of a 16th)

rng = np.random.default_rng(124)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ----------------------------------------------------------------------------
# DSP helpers
# ----------------------------------------------------------------------------
def mtof(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=float) - 69) / 12)


def rbj(kind, fc, q):
    fc = float(np.clip(fc, 10, SR * 0.45))
    w = 2 * np.pi * fc / SR
    cw, sw = np.cos(w), np.sin(w)
    al = sw / (2 * q)
    if kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
    elif kind == "bp":
        b = [al, 0, -al]
    else:
        raise ValueError(kind)
    a = [1 + al, -2 * cw, 1 - al]
    return np.array(b) / a[0], np.array(a) / a[0]


def filt(x, kind, fc, q=0.707):
    b, a = rbj(kind, fc, q)
    return signal.lfilter(b, a, x, axis=0)


def tv_filter(x, fc, q=0.707, kind="lp", block=32, stages=1):
    """Time-varying biquad (block-wise coefficient updates, state carried)."""
    y = np.asarray(x, dtype=float)
    for _ in range(stages):
        out = np.empty_like(y)
        zi = np.zeros(2)
        for i in range(0, len(y), block):
            b, a = rbj(kind, fc[i], q)
            out[i:i + block], zi = signal.lfilter(b, a, y[i:i + block], zi=zi)
        y = out
    return y


def polyblep(t, dt):
    out = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt[m]
    out[m] = x + x - x * x - 1
    m = t > 1 - dt
    x = (t[m] - 1) / dt[m]
    out[m] = x * x + x + x + 1
    return out


def saw_from_freq(freq, phase0=0.0):
    dt = np.maximum(freq / SR, 1e-9)
    t = (phase0 + np.cumsum(dt)) % 1.0
    return 2 * t - 1 - polyblep(t, dt), t, dt


def square_from_phase(t, dt):
    t2 = (t + 0.5) % 1.0
    s1 = 2 * t - 1 - polyblep(t, dt)
    s2 = 2 * t2 - 1 - polyblep(t2, dt)
    return s1 - s2


def pan(mono, p):
    """Constant-power pan, p in [-1, 1]."""
    a = (p + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], axis=1)


def stereo(x):
    return x if x.ndim == 2 else np.stack([x, x], axis=1)


def shift(x, n):
    y = np.zeros_like(x)
    if n < len(x):
        y[n:] = x[:len(x) - n]
    return y


def place(buf, snd, t, gain=1.0):
    i = int(round(t * SR))
    if i >= len(buf) or i < 0:
        return
    n = min(len(snd), len(buf) - i)
    buf[i:i + n] += snd[:n] * gain


def make_ir(rt60, predelay=0.02, damp=6000, hp=250, seed=0, length=None):
    r = np.random.default_rng(seed)
    length = length or rt60 * 1.2
    n = int(length * SR)
    t = np.arange(n) / SR
    env = np.exp(-6.91 * t / rt60)
    # slow fade-in of the diffuse tail = "bloom"
    env *= 1 - np.exp(-t / 0.015)
    ir = r.standard_normal((n, 2)) * env[:, None]
    # high frequencies decay faster: crossfade a darker version
    dark = filt(ir, "lp", damp * 0.35)
    mix = np.clip(t / rt60, 0, 1)[:, None]
    ir = ir * (1 - mix) + dark * mix
    ir = filt(ir, "lp", damp)
    ir = filt(ir, "hp", hp)
    pd = int(predelay * SR)
    ir = np.concatenate([np.zeros((pd, 2)), ir])
    return ir / np.sqrt(np.sum(ir ** 2) / 2)


def reverb(x, ir, wet=0.3):
    x = stereo(x)
    w = np.stack([signal.oaconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], axis=1)
    return w * wet


def pingpong(x, delay_s, fb=0.45, n=10, lp=3500, hp=300, wet=0.35):
    """Ping-pong delay with filtered feedback. Returns the wet signal (stereo)."""
    x = stereo(x)
    mono = x.mean(axis=1)
    d = int(delay_s * SR)
    out = np.zeros((len(mono), 2))
    tap = mono
    for k in range(1, n + 1):
        tap = filt(filt(shift(tap, d), "lp", lp), "hp", hp) * fb if k > 1 else \
            filt(filt(shift(tap, d), "lp", lp), "hp", hp)
        out[:, (k - 1) % 2] += tap
    return out * wet


def env_ad(n, a, d, sr=SR):
    t = np.arange(n) / sr
    e = np.where(t < a, t / max(a, 1e-6), np.exp(-(t - a) / d))
    return e


# ----------------------------------------------------------------------------
# Timeline / arrangement helpers
# ----------------------------------------------------------------------------
def bar_t(bar):
    return bar * BAR


def step_t(bar, step, swing=False):
    t = bar * BAR + step * STEP
    if swing and step % 2 == 1:
        t += SWING * STEP
    return t


def curve(keys, kind="log"):
    """Per-sample automation curve from [(bar, value), ...] keyframes."""
    bars = np.array([k[0] for k in keys], dtype=float) * BAR
    vals = np.array([k[1] for k in keys], dtype=float)
    t = np.arange(TOTAL) / SR
    if kind == "log":
        return np.exp(np.interp(t, bars, np.log(vals)))
    return np.interp(t, bars, vals)


def bars_in(*ranges):
    s = set()
    for a, b in ranges:
        s.update(range(a, b))
    return s


# G natural minor chord loop, 2 bars per chord: i - VI - iv - v  (Gm Eb Cm Dm)
# chord tones (MIDI) in a mid register: root, 3rd, 5th, 7th, 9th
CHORDS = [
    [55, 58, 62, 65, 69],   # Gm9  : G Bb D F A
    [51, 55, 58, 62, 65],   # Ebmaj9: Eb G Bb D F
    [48, 51, 55, 58, 62],   # Cm9  : C Eb G Bb D
    [50, 53, 57, 60, 64],   # Dm7add9 (natural minor v)
]
ROOTS = [31, 27, 24, 26]   # G1 Eb1 C1 D1 (bass plays +12 / +24)


def chord_at(bar):
    return (bar // 2) % 4


# --------------------------------------------------------------------------
# ARRANGEMENT (0-indexed bars)
#   0-15   intro: kick, closed hats from 8, rim
#   16-31  + rumble, shaker, clap, open hat
#   32-47  + rolling bass
#   48-63  + arp (filtered), dub stabs from 56
#   64-95  main groove: arp cutoff opens, ride, kick dropout end of 95
#   96-119 breakdown: kick/bass out, pad + arp blooms, riser + snare roll
#   120-159 drop: everything, crash/impact, arp variation 136-151
#   160-175 strip-down: arp closes & exits 168, stabs out 168
#   176-191 DJ outro: kick, rumble, hats, perc
# --------------------------------------------------------------------------
KICK_BARS = bars_in((0, 96), (120, 192))
RUMBLE_BARS = bars_in((16, 96), (120, 188))
CLOSED_HAT_BARS = bars_in((8, 96), (120, 184))
OPEN_HAT_BARS = bars_in((24, 95), (120, 176))
CLAP_BARS = bars_in((16, 96), (120, 184))
RIM_BARS = bars_in((0, 96), (104, 192))
SHAKER_BARS = bars_in((16, 96), (120, 180))
RIDE_BARS = bars_in((64, 95), (120, 160))
BASS_BARS = bars_in((32, 96), (120, 176))
ARP_BARS = bars_in((48, 168))
STAB_BARS = bars_in((56, 168))
PAD_BARS = bars_in((92, 124), (136, 160))


# ----------------------------------------------------------------------------
# Instruments
# ----------------------------------------------------------------------------
def kick_sound(dur=0.55):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 47 + 110 * np.exp(-t / 0.018) + 60 * np.exp(-t / 0.004)
    ph = 2 * np.pi * np.cumsum(f) / SR
    amp = np.where(t < 0.06, 1.0, np.exp(-(t - 0.06) / 0.16))
    body = np.sin(ph) * amp
    click = filt(rng.standard_normal(n), "hp", 3000) * np.exp(-t / 0.0025) * 0.35
    k = np.tanh(2.2 * (body + click)) / np.tanh(2.2)
    k *= np.minimum(1, (n - np.arange(n)) / (0.01 * SR))
    return k


def hat_sound(decay, open_=False):
    n = int((decay * 6) * SR)
    t = np.arange(n) / SR
    ratios = [2.0, 3.0, 4.16, 5.43, 6.79, 8.21]
    metal = sum(np.sign(np.sin(2 * np.pi * 320 * r * t + rng.uniform(0, 6))) for r in ratios)
    x = 0.5 * metal / 6 + 0.6 * rng.standard_normal(n)
    x = filt(filt(x, "bp", 10000, 0.9), "hp", 7000)
    return x * env_ad(n, 0.0005, decay)


def clap_sound():
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    e = np.zeros(n)
    for i, off in enumerate([0.0, 0.009, 0.019]):
        m = t >= off
        e[m] = np.maximum(e[m], np.exp(-(t[m] - off) / 0.006) * (0.8 + 0.1 * i))
    m = t >= 0.026
    e[m] = np.maximum(e[m], np.exp(-(t[m] - 0.026) / 0.09) * 0.7)
    x = filt(filt(noise, "bp", 1250, 1.2), "hp", 600) * e
    return x * 1.6


def snare_sound(pitch=1.0):
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 190 * pitch * t) * np.exp(-t / 0.05)
    nz = filt(rng.standard_normal(n), "bp", 2500 * pitch, 0.8) * np.exp(-t / 0.07)
    return 0.5 * tone + 0.9 * nz


def rim_sound(freq=820):
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.018)
    x += 0.6 * np.sin(2 * np.pi * freq * 2.37 * t) * np.exp(-t / 0.008)
    x += filt(rng.standard_normal(n), "bp", 3200, 2) * np.exp(-t / 0.004) * 0.8
    return np.tanh(1.5 * x)


def shaker_sound():
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    e = (1 - np.exp(-t / 0.008)) * np.exp(-t / 0.035)
    return filt(filt(rng.standard_normal(n), "bp", 6000, 1.5), "hp", 4000) * e * 2


def ride_sound():
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    ratios = [1.0, 1.48, 1.93, 2.53, 3.11, 3.79]
    x = sum(np.sin(2 * np.pi * 520 * r * t + rng.uniform(0, 6)) for r in ratios) / 6
    x = np.sign(x) * np.abs(x) ** 0.6
    x = filt(x + 0.3 * rng.standard_normal(n), "hp", 5000)
    return x * env_ad(n, 0.001, 0.35)


def crash_sound(dur=3.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = filt(rng.standard_normal(n), "hp", 4500)
    x += 0.4 * filt(rng.standard_normal(n), "bp", 9000, 0.7)
    return x * env_ad(n, 0.002, 0.7)


# ----------------------------------------------------------------------------
# Stems
# ----------------------------------------------------------------------------
def sidechain_shape(kick_times, release=0.42):
    """0..1 ducking shape that peaks at every kick and recovers over `release`."""
    d = np.zeros(TOTAL)
    n = int(release * SR)
    x = np.arange(n) / n
    shape = (1 - x) ** 2
    attack = int(0.002 * SR)
    shape[:attack] *= np.linspace(0, 1, attack)
    for kt in kick_times:
        i = int(round(kt * SR))
        m = min(n, TOTAL - i)
        d[i:i + m] = np.maximum(d[i:i + m], shape[:m])
    return d


def render_drums():
    out = {}
    kick = np.zeros(TOTAL)
    ks = kick_sound()
    kick_times = []
    for bar in KICK_BARS:
        for beat in range(4):
            # one-beat kick drop-out right before the drop and before the breakdown
            if bar in (95,) and beat >= 2:
                continue
            if bar == 63 and beat == 3:
                continue
            t = bar_t(bar) + beat * BEAT
            kick_times.append(t)
            place(kick, ks, t)
    out["kick"] = stereo(kick)

    # rumble: kick -> drive -> long dark reverb -> band-limit -> heavy sidechain
    send = np.zeros(TOTAL)
    for bar in RUMBLE_BARS:
        for beat in range(4):
            place(send, ks, bar_t(bar) + beat * BEAT)
    ir = make_ir(3.2, predelay=0.01, damp=900, hp=30, seed=3)
    rum = reverb(np.tanh(send * 2.5), ir, wet=1.0).mean(axis=1)
    rum = filt(filt(rum, "lp", 170, 0.9), "lp", 170, 0.7)
    rum = filt(rum, "hp", 45)
    rum = np.tanh(rum * 3.0)
    rum = filt(rum, "lp", 220)
    sc = sidechain_shape(kick_times, release=BEAT * 0.9)
    rum *= 1 - 0.97 * sc
    out["rumble"] = stereo(rum)

    # hats
    ch = np.zeros((TOTAL, 2))
    oh = np.zeros((TOTAL, 2))
    vel = [0.9, 0.42, 0.65, 0.45]
    for bar in CLOSED_HAT_BARS:
        for s in range(16):
            if s % 4 == 2 and bar in OPEN_HAT_BARS:
                continue
            v = vel[s % 4] * rng.uniform(0.85, 1.15)
            dec = 0.018 + 0.012 * (0.5 + 0.5 * np.sin(2 * np.pi * bar / 16))
            snd = pan(hat_sound(dec), 0.25)
            place(ch, snd, step_t(bar, s, swing=True), v)
    for bar in OPEN_HAT_BARS:
        for beat in range(4):
            snd = pan(hat_sound(0.085, True), -0.15)
            place(oh, snd, step_t(bar, beat * 4 + 2), rng.uniform(0.8, 1.0))
    out["hats"] = ch * 0.55 + oh * 0.8

    # clap on 2 & 4 + plate reverb
    clap = np.zeros(TOTAL)
    cs = clap_sound()
    for bar in CLAP_BARS:
        for beat in (1, 3):
            place(clap, cs, bar_t(bar) + beat * BEAT + 0.003, rng.uniform(0.9, 1.0))
    # snare roll in the build (bars 112-119)
    roll = np.zeros(TOTAL)
    for bar in range(112, 120):
        div = 4 if bar < 114 else 8 if bar < 116 else 16 if bar < 119 else 32
        n = div
        for i in range(n):
            t = bar_t(bar) + i * BAR / n
            prog = ((bar - 112) * BAR + i * BAR / n) / (8 * BAR)
            place(roll, snare_sound(1 + 0.6 * prog), t, 0.2 + 0.8 * prog ** 1.5)
    plate = make_ir(1.3, predelay=0.015, damp=7000, hp=350, seed=5)
    clapst = stereo(clap) + reverb(clap, plate, wet=0.35)
    rollst = stereo(roll) + reverb(roll, make_ir(2.5, seed=6), wet=0.4)
    out["clap"] = clapst
    out["snare_roll"] = rollst

    # rim / perc: 3-against-4 minimal loop with variation, swing, panned
    perc = np.zeros((TOTAL, 2))
    pat_a = [0, 3, 6, 10, 13]
    pat_b = [2, 5, 8, 11, 14]
    for bar in RIM_BARS:
        pat = pat_a if (bar // 4) % 2 == 0 else pat_b
        for j, s in enumerate(pat):
            f = 820 if j % 2 == 0 else 640
            snd = pan(rim_sound(f), -0.35 if j % 2 == 0 else 0.35)
            v = (0.55 if bar < 16 else 0.8) * rng.uniform(0.8, 1.0)
            place(perc, snd, step_t(bar, s, swing=True), v)
        # low conga-ish tom every 4 bars
        if bar % 4 == 3:
            place(perc, pan(rim_sound(210) * 1.2, 0.1), step_t(bar, 14, True), 0.7)
    perc += pingpong(perc, STEP * 3, fb=0.4, n=6, lp=4000, hp=500, wet=0.3)
    out["perc"] = perc

    shaker = np.zeros((TOTAL, 2))
    for bar in SHAKER_BARS:
        for s in range(16):
            v = (0.9 if s % 4 == 2 else 0.5) * rng.uniform(0.8, 1.1)
            place(shaker, pan(shaker_sound(), -0.4), step_t(bar, s, swing=True), v)
    out["shaker"] = shaker

    ride = np.zeros((TOTAL, 2))
    rs = pan(ride_sound(), 0.3)
    for bar in RIDE_BARS:
        for beat in range(4):
            place(ride, rs, step_t(bar, beat * 4 + 2), 0.6)
    out["ride"] = ride
    return out, kick_times


def mono_synth(notes, cutoff, q, wave_mix, f_env_amt, f_env_dec, amp_dec,
               gate, sub=0.0, drive=1.0, detune=0.0, stages=2):
    """notes: list of (time_s, midi, vel, accent)."""
    freq = np.full(TOTAL, mtof(notes[0][1]) if notes else 100.0)
    amp = np.zeros(TOTAL)
    fenv = np.zeros(TOTAL)
    notes = sorted(notes)
    gate_n = int(gate * SR)
    rel = int(0.006 * SR)
    for idx, (t, m, vel, acc) in enumerate(notes):
        i = int(round(t * SR))
        j = int(round(notes[idx + 1][0] * SR)) if idx + 1 < len(notes) else TOTAL
        freq[i:j] = mtof(m)
        n = min(gate_n + rel, TOTAL - i)
        tt = np.arange(n) / SR
        e = np.minimum(1, tt / 0.002) * np.exp(-tt / amp_dec)
        e[gate_n:] *= np.linspace(1, 0, n - gate_n) if n > gate_n else 1
        amp[i:i + n] = np.maximum(amp[i:i + n], e * vel)
        nf = min(int(0.6 * SR), TOTAL - i)
        tf = np.arange(nf) / SR
        fe = np.exp(-tf / f_env_dec) * f_env_amt * (1 + 1.2 * acc) * vel
        fenv[i:i + nf] = np.maximum(fenv[i:i + nf], fe)
    saw, ph, dt = saw_from_freq(freq)
    sq = square_from_phase(ph, dt)
    osc = wave_mix * saw + (1 - wave_mix) * sq
    if detune:
        saw2, _, _ = saw_from_freq(freq * 2 ** (detune / 1200), 0.37)
        osc = 0.6 * osc + 0.4 * saw2
    fc = np.clip(cutoff * (1 + fenv), 30, 16000)
    y = tv_filter(osc, fc, q=q, kind="lp", stages=stages)
    y = np.tanh(drive * y) / np.tanh(drive)
    if sub:
        y += sub * np.sin(2 * np.pi * np.cumsum(freq / 2) / SR)
    return y * amp


def render_bass(kick_times):
    notes = []
    for bar in BASS_BARS:
        root = ROOTS[chord_at(bar)] + 12
        for beat in range(4):
            for s, (iv, v) in zip((1, 2, 3), ((0, 0.75), (12, 1.0), (0, 0.8))):
                # occasional fifth instead of the octave for movement
                if s == 2 and beat == 3 and bar % 2 == 1:
                    iv = 7
                notes.append((step_t(bar, beat * 4 + s), root + iv, v, 1.0 if s == 2 else 0.0))
    cutoff = curve([(0, 180), (32, 180), (64, 260), (96, 300), (120, 320), (176, 320)])
    y = mono_synth(notes, cutoff, q=1.6, wave_mix=0.8, f_env_amt=3.0, f_env_dec=0.05,
                   amp_dec=0.12, gate=STEP * 0.8, sub=0.0, drive=2.0, stages=2)
    # sub layer (pure sine on the same notes, lowpassed)
    y = filt(y, "hp", 38)
    sc = sidechain_shape(kick_times, release=BEAT * 0.45)
    y *= 1 - 0.6 * sc
    return stereo(y)


ARP_A = [0, 4, 2, 4, 1, 4, 2, 3, 0, 4, 2, 4, 1, 5, 3, 2]
ARP_B = [4, 0, 5, 2, 6, 1, 4, 3, 5, 0, 6, 2, 4, 1, 3, 5]


def arp_tones(c):
    ch = CHORDS[c]
    return ch + [ch[0] + 12, ch[2] + 12]   # 7 tones


def render_arp(kick_times):
    notes = []
    for bar in ARP_BARS:
        tones = arp_tones(chord_at(bar))
        pat = ARP_B if 136 <= bar < 152 else ARP_A
        for s in range(16):
            m = tones[pat[s]] + 12 * (1 if (bar >= 104 and s % 8 == 7) else 0)
            acc = 1.0 if s % 3 == 0 else 0.0           # 3-against-4 accents
            notes.append((step_t(bar, s), m, 0.75 + 0.25 * acc, acc))
    cutoff = curve([(0, 250), (48, 250), (56, 420), (64, 600), (80, 1100), (95, 1700),
                    (96, 500), (104, 900), (112, 2200), (119, 4200), (120, 2600),
                    (136, 3600), (152, 2600), (160, 1200), (168, 350), (192, 350)])
    # slow LFO on cutoff (8 bar period) for movement
    t = np.arange(TOTAL) / SR
    cutoff = cutoff * (1 + 0.25 * np.sin(2 * np.pi * t / (8 * BAR)))
    y = mono_synth(notes, cutoff, q=4.2, wave_mix=0.55, f_env_amt=2.2, f_env_dec=0.09,
                   amp_dec=0.22, gate=STEP * 0.6, drive=1.8, detune=9, stages=2)
    y = filt(y, "hp", 160)
    sc = sidechain_shape(kick_times, release=BEAT * 0.5)
    y *= 1 - 0.35 * sc
    dry = np.stack([y, shift(y, int(0.012 * SR)) * 0.92], axis=1)   # Haas width
    wet_amt = curve([(0, 0.25), (96, 0.25), (100, 0.4), (119, 0.45), (120, 0.25), (192, 0.25)], "lin")
    dly = pingpong(y, STEP * 3, fb=0.42, n=8, lp=3000, hp=400, wet=1.0)
    hall = make_ir(5.0, predelay=0.03, damp=5000, hp=300, seed=9)
    rv = reverb(y, hall, wet=1.0)
    return dry + dly * (0.35 * wet_amt / 0.25)[:, None] * 0.6 + rv * wet_amt[:, None]


def poly_note(midi, dur, voices=3, spread=12, cutoff=1500, q=0.9, a=0.003, d=0.12,
              sustain_shape=False, seed=0):
    n = int((dur + (d * 5 if not sustain_shape else 1.5)) * SR)
    r = np.random.default_rng(seed)
    f = mtof(midi)
    out = np.zeros((n, 2))
    for v in range(voices):
        cents = (v - (voices - 1) / 2) * spread / max(1, (voices - 1) / 2)
        fr = np.full(n, f * 2 ** (cents / 1200))
        s, _, _ = saw_from_freq(fr, r.uniform())
        p = (v / max(1, voices - 1)) * 2 - 1 if voices > 1 else 0
        out += pan(s, p * 0.7)
    t = np.arange(n) / SR
    if sustain_shape:
        e = np.minimum(1, t / a) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.8))
    else:
        e = np.minimum(1, t / a) * np.exp(-t / d)
    out = filt(out, "lp", cutoff, q)
    return out * e[:, None] / voices


def render_stabs(kick_times):
    dry = np.zeros((TOTAL, 2))
    cache = {}
    pattern = [3, 10]        # syncopated offbeat 16ths
    for bar in STAB_BARS:
        c = chord_at(bar)
        steps = pattern + ([14] if bar % 4 == 3 else [])
        lvl = 0.6 if bar < 64 else 1.0
        for s in steps:
            key = (c, s == 14)
            if key not in cache:
                snd = sum(poly_note(m, 0.05, voices=3, spread=10, cutoff=1400, q=1.4,
                                    d=0.07, seed=m) for m in CHORDS[c][:4])
                cache[key] = snd * (0.8 if s == 14 else 1.0)
            place(dry, cache[key], step_t(bar, s), lvl)
    dry = filt(dry, "hp", 180)
    # dub delay (3/16) with dark feedback + reverb: the dub-techno chord trail
    dly = pingpong(dry, STEP * 3, fb=0.55, n=12, lp=1800, hp=350, wet=0.8)
    hall = make_ir(3.5, predelay=0.025, damp=4000, hp=300, seed=11)
    rv = reverb(dry + dly * 0.5, hall, wet=0.45)
    y = dry + dly + rv
    sc = sidechain_shape(kick_times, release=BEAT * 0.6)
    return y * (1 - 0.45 * sc)[:, None]


def render_pad(kick_times):
    out = np.zeros((TOTAL, 2))
    for bar in sorted(PAD_BARS):
        if bar % 2:
            continue
        c = chord_at(bar)
        notes = CHORDS[c][:4] + [CHORDS[c][0] - 12]
        for m in notes:
            snd = poly_note(m, BAR * 2, voices=5, spread=18, cutoff=1600, q=0.8,
                            a=0.6, sustain_shape=True, seed=m + bar)
            place(out, snd, bar_t(bar), 0.5)
    # slow breathing filter
    t = np.arange(TOTAL) / SR
    fc = 900 * (1 + 0.6 * np.sin(2 * np.pi * t / (4 * BAR)))
    out = np.stack([tv_filter(out[:, ch], fc, 0.7, block=256) for ch in range(2)], axis=1)
    out = filt(out, "hp", 200)
    hall = make_ir(6.0, predelay=0.04, damp=4500, hp=300, seed=13)
    y = out + reverb(out, hall, wet=0.8)
    # fade in/out pad region edges with a gain curve
    g = curve([(0, 0), (92, 0), (100, 1), (119, 1.1), (120, 0.6), (124, 0), (136, 0),
               (140, 0.5), (156, 0.5), (160, 0), (192, 0)], "lin")
    sc = sidechain_shape(kick_times, release=BEAT * 0.8)
    return y * (g * (1 - 0.5 * sc))[:, None]


def render_fx():
    fx = np.zeros((TOTAL, 2))
    # risers: 8 bars before the drop, 2 bars before breakdown/main sections
    for start, length, level in ((112, 8, 1.0), (62, 2, 0.5), (94, 2, 0.5), (158, 2, 0.4)):
        n = int(length * BAR * SR)
        t = np.arange(n) / SR
        prog = t / t[-1]
        noise = rng.standard_normal((n, 2))
        fc = 200 * (8000 / 200) ** prog
        nz = np.stack([tv_filter(noise[:, c], fc, 1.2, "bp", block=64) for c in range(2)], axis=1)
        sweep_f = mtof(55) * 2 ** (2 * prog ** 2)
        tone, _, _ = saw_from_freq(sweep_f)
        tone = filt(tone, "lp", 3000)
        r = (nz * 0.8 + stereo(tone) * 0.12) * (prog ** 2.2)[:, None]
        r *= np.minimum(1, (n - np.arange(n)) / (0.02 * SR))[:, None]
        place(fx, r, bar_t(start), level)
    # crash + sub-drop impact at drops
    crash = pan(crash_sound(), 0)
    crash_w = crash + reverb(crash, make_ir(3.0, seed=17), 0.4)
    for bar, lvl in ((64, 0.6), (120, 1.0), (136, 0.6), (152, 0.6)):
        place(fx, crash_w, bar_t(bar), lvl)
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(60 * np.exp(-t / 0.9) + 28) / SR) * np.exp(-t / 0.7)
    place(fx, stereo(boom), bar_t(120), 0.7)
    # reverse cymbal into the breakdown and into bar 64
    rev = crash[::-1][-int(BAR * SR):]
    for bar in (63, 95, 119):
        place(fx, rev, bar_t(bar + 1) - len(rev) / SR, 0.5)
    return fx


# ----------------------------------------------------------------------------
# Mix & master
# ----------------------------------------------------------------------------
# Target RMS (dB, relative to the kick) measured over each stem's reference
# section: (target_db, (start_bar, end_bar)).  Auto-gain makes balancing robust.
MIX = {
    "kick": (0.0, (128, 136)),
    "rumble": (-9.0, (128, 136)),
    "bass": (-5.5, (128, 136)),
    "hats": (-13.0, (128, 136)),
    "clap": (-13.5, (128, 136)),
    "perc": (-15.0, (128, 136)),
    "shaker": (-21.0, (128, 136)),
    "ride": (-22.0, (128, 136)),
    "arp": (-7.5, (128, 136)),
    "stabs": (-13.0, (128, 136)),
    "pad": (-14.0, (104, 118)),
    "snare_roll": (-10.0, (116, 120)),
    "fx": (-12.0, (116, 120)),
}


# Mix-stage level automation (linear gain keyframes per stem).
AUTOMATION = {
    "arp": [(0, 1.0), (96, 1.0), (96.25, 0.55), (112, 0.6), (119.75, 0.85), (120, 1.0)],
}


def rms_db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-20)


def master(x):
    x = filt(x, "hp", 25)
    # gentle glue compression (RMS detector, 2:1 above threshold)
    det = np.sqrt(np.maximum(filt(np.mean(x ** 2, axis=1), "lp", 8), 0) + 1e-12)
    thr = 0.25
    gain = np.where(det > thr, (det / thr) ** (1 / 2 - 1), 1.0)
    x = x * gain[:, None]
    # normalise then lookahead peak limiter + soft clip
    x = x / np.max(np.abs(x)) * 2.6
    ceiling = 0.89
    look = int(0.003 * SR)
    peak = maximum_filter1d(np.max(np.abs(x), axis=1), size=2 * look + 1)
    g = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    g = minimum_filter1d(g, size=2 * look + 1)
    g = signal.filtfilt(*signal.butter(1, 30 / (SR / 2)), g)
    g = np.minimum(g, 1.0)
    x = x * g[:, None]
    x = np.tanh(x / ceiling) * ceiling * 1.0
    # true-peak ceiling: -1.2 dBTP measured with 4x oversampling
    tp = np.max(np.abs(signal.resample_poly(x, 4, 1, axis=0)))
    return x / tp * 10 ** (-1.2 / 20)


def write_wav(path, x):
    x = stereo(np.asarray(x))
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def render_all():
    print("drums...")
    stems, kick_times = render_drums()
    print("bass...")
    stems["bass"] = render_bass(kick_times)
    print("arp...")
    stems["arp"] = render_arp(kick_times)
    print("stabs...")
    stems["stabs"] = render_stabs(kick_times)
    print("pad...")
    stems["pad"] = render_pad(kick_times)
    print("fx...")
    stems["fx"] = render_fx()
    return {k: stereo(v).astype(np.float32) for k, v in stems.items()}


def main():
    import sys
    os.makedirs(os.path.join(OUT, "loops"), exist_ok=True)
    cache = os.path.join(OUT, ".stem_cache.npz")
    if "--mix-only" in sys.argv and os.path.exists(cache):
        stems = dict(np.load(cache))
    else:
        stems = render_all()
        np.savez(cache, **stems)

    def seg(v, bars):
        return v[int(bar_t(bars[0]) * SR):int(bar_t(bars[1]) * SR)]

    kick_ref = rms_db(seg(stems["kick"], MIX["kick"][1]))
    mix = np.zeros((TOTAL, 2))
    for k in MIX:
        target, ref = MIX[k]
        g = 10 ** ((kick_ref + target - rms_db(seg(stems[k], ref))) / 20)
        stems[k] = stems[k].astype(float) * g
        if k in AUTOMATION:
            stems[k] *= curve(AUTOMATION[k], "lin")[:, None]
        mix += stems[k]
        print(f"  {k:11s} gain {20*np.log10(g):6.1f} dB  peak "
              f"{20*np.log10(np.max(np.abs(stems[k]))+1e-9):6.1f} dBFS")
    pre_peak = np.max(np.abs(mix))
    print(f"pre-master peak {20*np.log10(pre_peak):.1f} dBFS")
    m = master(mix)

    fade = int(TAIL * SR)
    m[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2

    full_wav = os.path.join(OUT, "low_orbit_full_mix.wav")
    write_wav(full_wav, m)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", full_wav, "-b:a", "320k",
                    os.path.join(OUT, "low_orbit_full_mix.mp3")], check=True)

    # 8-bar stem loops from the main drop (bars 128-135). One common gain so
    # the stems sum back to (roughly) the un-limited mix with headroom.
    a, b = int(bar_t(128) * SR), int(bar_t(136) * SR)
    loop_gain = 0.7 / np.max(np.abs(mix[a:b]))
    for k, v in stems.items():
        sg = v[a:b] * loop_gain
        if np.max(np.abs(sg)) < 1e-4:
            continue
        write_wav(os.path.join(OUT, "loops", f"{k}_8bar.wav"), sg)
    write_wav(os.path.join(OUT, "loops", "full_loop_8bar.wav"), m[a:b])
    a2, b2 = int(bar_t(104) * SR), int(bar_t(112) * SR)
    write_wav(os.path.join(OUT, "loops", "breakdown_loop_8bar.wav"), m[a2:b2])
    print("done ->", OUT)


if __name__ == "__main__":
    main()
