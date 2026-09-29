"""Synthesize the sizzle soundtrack: 30 s of original corporate techno, scored to the cut.

Everything is generated here from oscillators and noise, so there is nothing to license.

    python music.py                 # writes music.wav
    python music.py --mux IN OUT    # also muxes it onto a video (video stream copied)

120 BPM, A minor (Am - F - C - G). The beat grid starts at 0.4 s so the drop lands on
the first cut (3.4 s), the sensor failure (~21.4 s) and the end card (27.4 s) fall on
downbeats, and the emergency-braking hit (9.9 s) falls on a beat.
"""

import argparse
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy import signal

SR = 44100
DUR = 30.0
BPM = 120
BEAT = 60 / BPM
GRID = 0.4                     # time of beat 0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(128)

# Cue points (seconds), matching sizzle.html / export.py
DROP, AEB, AEB_BACK, BLIND, BLIND_BACK, ARCH, END = 3.4, 9.9, 11.4, 21.4, 24.4, 24.4, 27.4

CHORDS = {  # MIDI notes
    "Am": [57, 60, 64], "F": [53, 57, 60], "C": [52, 55, 60], "G": [55, 59, 62],
}
PROG = ["Am", "F", "C", "G"]
ROOTS = {"Am": 45, "F": 41, "C": 48, "G": 43}


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def beat_time(b):
    return GRID + b * BEAT


def chord_at(time):
    bar = int(np.floor((time - GRID) / (4 * BEAT)))
    return PROG[bar % 4]


def place(buf, sound, at, gain=1.0):
    i = int(round(at * SR))
    if i >= len(buf) or i + len(sound) <= 0:
        return
    s0 = max(0, -i)
    j = min(len(buf), i + len(sound))
    buf[max(i, 0):j] += gain * sound[s0:s0 + j - max(i, 0)]


def env(n, attack, decay):
    k = np.arange(n) / SR
    a = np.clip(k / max(attack, 1e-4), 0, 1)
    return a * np.exp(-k / decay)


def saw(f, n, detune_cents=(0,), phase_seed=0):
    k = np.arange(n) / SR
    out = np.zeros(n)
    r = np.random.default_rng(phase_seed)
    for c in detune_cents:
        ff = f * 2 ** (c / 1200)
        out += 2 * ((k * ff + r.random()) % 1.0) - 1
    return out / len(detune_cents)


def lowpass(x, cutoff, q=0.707):
    b, a = signal.iirfilter(2, min(cutoff, SR * 0.45) / (SR / 2), btype="low", ftype="butter")
    return signal.lfilter(b, a, x)


def sweep_filter(x, cutoffs, btype="low", block=1024):
    """Time-varying 2-pole filter, cutoff given per sample; state carried across blocks."""
    y = np.zeros_like(x)
    zi = np.zeros(2)
    for s in range(0, len(x), block):
        fc = float(np.clip(cutoffs[min(s + block // 2, len(x) - 1)], 30, SR * 0.45))
        b, a = signal.butter(2, fc / (SR / 2), btype=btype)
        y[s:s + block], zi = signal.lfilter(b, a, x[s:s + block], zi=zi)
    return y


def ramp(t0, t1, v0, v1, curve=1.0):
    u = np.clip((t - t0) / (t1 - t0), 0, 1) ** curve
    return v0 + (v1 - v0) * u


# ---------------------------------------------------------------- instruments

def kick():
    n = int(0.5 * SR)
    k = np.arange(n) / SR
    f = 45 + 110 * np.exp(-k / 0.035)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-k / 0.28)
    click = rng.standard_normal(n) * np.exp(-k / 0.002) * 0.3
    return np.tanh(1.6 * (body + click))


def clap():
    n = int(0.35 * SR)
    k = np.arange(n) / SR
    noise = rng.standard_normal(n)
    e = np.zeros(n)
    for d in (0, 0.011, 0.022):
        e += np.where(k >= d, np.exp(-(k - d) / 0.008), 0) * (1 if d < 0.02 else 0)
    e += np.where(k >= 0.022, np.exp(-(k - 0.022) / 0.11), 0)
    b, a = signal.butter(2, [900 / (SR / 2), 3200 / (SR / 2)], btype="band")
    return signal.lfilter(b, a, noise * e) * 2.2


def hat(open_=False):
    n = int((0.25 if open_ else 0.06) * SR)
    k = np.arange(n) / SR
    x = rng.standard_normal(n) * np.exp(-k / (0.07 if open_ else 0.014))
    b, a = signal.butter(2, 7500 / (SR / 2), btype="high")
    return signal.lfilter(b, a, x)


def bass_note(m, length):
    n = int(length * SR)
    x = saw(hz(m), n, (-6, 6), m) + 0.5 * np.sin(2 * np.pi * hz(m - 12) * np.arange(n) / SR)
    k = np.arange(n) / SR
    y = sweep_filter(x, 180 + 1400 * np.exp(-k / 0.06))
    return y * env(n, 0.003, 0.16)


def pluck(m, length=0.22, bright=1.0):
    n = int(length * SR)
    k = np.arange(n) / SR
    x = saw(hz(m), n, (-8, 0, 8), m + 3)
    y = sweep_filter(x, 500 + 5200 * bright * np.exp(-k / 0.05))
    return y * env(n, 0.002, 0.09)


def pad_chord(notes, length):
    n = int(length * SR)
    x = sum(saw(hz(m), n, (-14, -5, 5, 14), m) for m in notes) / len(notes)
    x += 0.4 * saw(hz(notes[0] - 12), n, (-7, 7), 99)
    k = np.arange(n) / SR
    fade = np.clip(k / 0.05, 0, 1) * np.clip((length - k) / 0.08, 0, 1)
    return x * fade


def riser(t0, t1):
    n = int((t1 - t0) * SR)
    k = np.arange(n) / SR
    u = k / (t1 - t0)
    x = rng.standard_normal(n)
    y = sweep_filter(x, 400 + 9000 * u ** 2, btype="band" if False else "low")
    tone = np.sin(2 * np.pi * np.cumsum(220 + 660 * u ** 2) / SR) * 0.25
    return (y * 0.7 + tone) * u ** 1.5


def impact():
    n = int(2.2 * SR)
    k = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(38 + 60 * np.exp(-k / 0.08)) / SR) * np.exp(-k / 0.6)
    noise = lowpass(rng.standard_normal(n), 2500) * np.exp(-k / 0.35) * 0.5
    return np.tanh(1.3 * (boom + noise))


def reverse_swell(length=0.9):
    n = int(length * SR)
    k = np.arange(n) / SR
    x = lowpass(rng.standard_normal(n), 6000) * (k / length) ** 3
    return x


# ------------------------------------------------------------------- arrange

def build():
    drums, bass, pads, arps, fx = (np.zeros(N) for _ in range(5))
    kicks = []

    groove_on = lambda s: (DROP <= s < AEB) or (AEB_BACK <= s < END)  # noqa: E731

    # beats on the grid
    b = 0
    while beat_time(b) < DUR:
        bt = beat_time(b)
        in_bar = b % 4
        if groove_on(bt):
            k_gain = 0.55 if BLIND <= bt < BLIND_BACK else 1.0
            place(drums, kick(), bt, 0.95 * k_gain)
            kicks.append(bt)
            if in_bar in (1, 3):
                place(drums, clap(), bt, 0.45)
            place(drums, hat(open_=True), bt + BEAT / 2, 0.26)
            for q in (0.25, 0.75):
                place(drums, hat(), bt + q * BEAT, 0.15)
            # rolling offbeat bass: three 16ths after each kick
            root = ROOTS[chord_at(bt)]
            for q, oct_ in ((0.25, 0), (0.5, 12), (0.75, 0)):
                place(bass, bass_note(root + oct_, BEAT / 4 * 0.95), bt + q * BEAT, 0.55)
        elif bt < DROP and bt >= 1.4:
            place(drums, hat(), bt + BEAT / 2, 0.07)          # intro ticks
        elif AEB <= bt < AEB_BACK:
            place(drums, hat(), bt, 0.08)                     # tension: clock ticks
            place(bass, bass_note(ROOTS["Am"], BEAT * 0.9), bt, 0.35)
        b += 1

    # snare/clap rolls into the drop, back from AEB, and out of the blind section
    for t0, t1 in ((2.4, DROP - 0.06), (AEB_BACK - BEAT, AEB_BACK - 0.03), (BLIND_BACK - 2 * BEAT, BLIND_BACK - 0.03)):
        steps = int((t1 - t0) / (BEAT / 4))
        for i in range(steps):
            u = i / max(1, steps - 1)
            place(drums, clap(), t0 + i * BEAT / 4, 0.12 + 0.3 * u)

    # pads: one chord per bar from the start (intro Am, F), sidechained later
    bar = -2
    while True:
        bt = GRID + bar * 4 * BEAT
        if bt >= DUR:
            break
        name = PROG[bar % 4]
        start, length = max(0.0, bt), 4 * BEAT + (bt if bt < 0 else 0)
        if length > 0 and start < END:
            place(pads, pad_chord(CHORDS[name], min(length, END - start)), start, 0.36)
        bar += 1
    place(pads, pad_chord([57, 60, 64, 69], DUR - END), END, 0.36)   # final Am, rings out

    # arps: 16ths through chord tones, from the second bar after the drop
    arp_start = DROP + 4 * BEAT
    s = arp_start
    i = 0
    while s < END:
        if not (AEB <= s < AEB_BACK):
            notes = CHORDS[chord_at(s)]
            pattern = [0, 1, 2, 1, 2, 0, 2, 1]
            m = notes[pattern[i % 8]] + 12 + (12 if (i % 16) in (6, 14) else 0)
            bright = 1.4 if s >= ARCH else 1.0
            place(arps, pluck(m, bright=bright), s, 0.3 if s >= 13.0 else 0.22)
        s += BEAT / 4
        i += 1
    # a bright lead motif on the architecture card
    for j, (m, dur) in enumerate([(76, 0.5), (74, 0.25), (72, 0.25), (76, 0.75), (79, 0.75), (81, 1.0)]):
        at = ARCH + sum(d for _, d in [(76, 0.5), (74, 0.25), (72, 0.25), (76, 0.75), (79, 0.75), (81, 1.0)][:j]) * BEAT * 2
        place(arps, pluck(m, length=dur * BEAT * 2, bright=1.6) * 1.0, at, 0.2)

    # fx
    place(fx, riser(1.2, DROP), 1.2, 0.5)
    place(fx, riser(BLIND_BACK - 2.0, BLIND_BACK), BLIND_BACK - 2.0, 0.45)
    place(fx, reverse_swell(0.9), END - 0.9, 0.5)
    for at, g in ((DROP, 0.8), (AEB, 1.0), (13.0, 0.35), (16.0, 0.3), (BLIND, 0.55), (ARCH, 0.5), (END, 1.0)):
        place(fx, impact(), at, g)

    # sidechain: duck pads, bass tails and arps on every kick
    duck = np.ones(N)
    for kt in kicks:
        i0 = int(kt * SR)
        n = min(int(0.4 * SR), N - i0)
        k = np.arange(n) / SR
        duck[i0:i0 + n] = np.minimum(duck[i0:i0 + n], 1 - 0.7 * np.exp(-k / 0.11))
    pads *= duck
    arps *= 0.6 + 0.4 * duck

    # pad filter: opens through the intro, dips during AEB and the blind section
    pad_cut = ramp(0, DROP, 350, 2600, 2) * np.where((t >= AEB) & (t < AEB_BACK), 0.35, 1.0)
    pads = sweep_filter(pads, pad_cut)

    music = drums + bass + pads + arps
    # sensor failure: the whole mix drops behind a low-pass, with stutter gating
    blind_cut = np.where((t >= BLIND) & (t < BLIND_BACK), ramp(BLIND, BLIND_BACK, 700, 9000, 3), 16000)
    music = sweep_filter(music, blind_cut)
    gate = np.ones(N)
    stutter = (t >= BLIND) & (t < BLIND + 1.0)
    gate[stutter] = (np.floor((t[stutter] - BLIND) / (BEAT / 4)) % 2 == 0) * 0.85 + 0.15
    music *= gate

    mix = music + fx
    # end: everything but pad/fx stops at END; fade the tail
    mix *= np.clip((DUR - t) / 1.4, 0, 1)
    return mix, pads


def stereo_and_master(mono):
    # short synthetic room reverb for width
    ir_len = int(1.6 * SR)
    k = np.arange(ir_len) / SR
    irL = rng.standard_normal(ir_len) * np.exp(-k / 0.35)
    irR = rng.standard_normal(ir_len) * np.exp(-k / 0.35)
    wetL = signal.fftconvolve(mono, irL)[:N] * 0.018
    wetR = signal.fftconvolve(mono, irR)[:N] * 0.018
    L, R = mono + wetL, mono + wetR
    st = np.stack([L, R], axis=1)
    st = signal.lfilter(*signal.butter(2, 28 / (SR / 2), btype="high"), st, axis=0)   # remove DC/rumble
    st = np.tanh(st / np.max(np.abs(st)) * 1.8) / np.tanh(1.8)                      # glue + limit
    st *= 10 ** (-1.0 / 20) / np.max(np.abs(st))                                      # -1 dBFS peak
    fade_in = np.clip(t / 0.02, 0, 1)[:, None]
    return st * fade_in


def write_wav(path, st):
    data = (st * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def mux(video, audio, out):
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
                    "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
                   check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("music.wav")))
    ap.add_argument("--mux", nargs=2, metavar=("VIDEO_IN", "VIDEO_OUT"))
    args = ap.parse_args()
    mono, _ = build()
    st = stereo_and_master(mono)
    write_wav(args.out, st)
    rms = 20 * np.log10(np.sqrt(np.mean(st ** 2)))
    print(f"wrote {args.out}: {DUR:.0f} s, RMS {rms:.1f} dBFS")
    if args.mux:
        mux(args.mux[0], args.out, args.mux[1])
        print(f"muxed -> {args.mux[1]}")


if __name__ == "__main__":
    main()
