# Low Orbit: minimal / progressive techno

This is an original track built entirely from code by `generate.py`, using numpy and scipy synthesis with no samples. It draws on melodic and minimal techno: Stephan Bodzin-style resonant sequenced arps, Afterlife/Tale Of Us-style breakdowns, rolling Berlin bass and rumble, and dub-techno chord stabs.

| | |
|---|---|
| Tempo | **124 BPM** |
| Key | **G minor** (chord loop Gm9 – Ebmaj9 – Cm9 – Dm7, 2 bars each) |
| Length | 192 bars, about 6:12 |
| Master | about −9.8 LUFS integrated, true peak ≤ −1.2 dBTP |

## Files

- `output/low_orbit_full_mix.mp3`: the full arranged and mastered track (320 kbps).
- `output/loops/*_8bar.wav`: 8-bar stems from the main drop (bars 128–135), 16-bit/44.1 kHz WAV, 15.48 s, so they loop seamlessly at 124 BPM. They share a common gain, so dropping them all into a DAW at 0 dB rebuilds the groove with headroom.
  - `kick`, `rumble`, `bass`, `hats`, `clap`, `perc`, `shaker`, `ride`, `arp`, `stabs`
- `output/loops/full_loop_8bar.wav`: the mastered 8-bar drop loop.
- `output/loops/breakdown_loop_8bar.wav`: the mastered 8-bar breakdown (arp, pad, dub stabs, perc).

## Sound design

- **Kick:** a sine swept from about 200 Hz to 47 Hz, with a 60 ms hold and 160 ms decay, a 2.5 ms noise click, and tanh saturation.
- **Rumble:** the kick goes through drive into a 3.2 s dark reverb, then a 45–220 Hz band, then 97% sidechain ducking.
- **Rolling bass:** a "0 1 1 1" 16th pattern of root, octave and root. It uses polyBLEP saw/square oscillators into a 2-stage resonant low-pass with a 50 ms filter envelope, plus drive and about 8 dB of sidechain.
- **Arp:** a monophonic 16th sequence over the chord tones, with 3-against-4 accents that hit the filter envelope harder.
  - Filter: a resonant 24 dB low-pass (Q 4.2), with its cutoff automated from 250 Hz to 4.2 kHz across the track plus an 8-bar LFO.
  - Effects: a 3/16 ping-pong delay, a 5 s hall and a 12 ms Haas widening.
- **Dub stabs:** min9/maj9 voicings from 3-voice detuned saws, played on 16th offbeats. They feed a 3/16 delay with dark feedback and a 3.5 s hall.
- **Pad:** 5-voice supersaw chords with a breathing filter and a 6 s hall, used in the breakdown and the second half of the drop.
- **Drums:**
  - Hats: 808-style metallic hats with 54% swing, 16th closed hats with velocity variation, and open hats on the offbeats.
  - Clap: a triple-hit clap with a plate reverb.
  - Rims: 3-against-4 rim/clave loops with a dotted delay.
  - Also a shaker, a ride, a snare-roll build, a riser (HPF 200 Hz → 8 kHz), a crash, a sub-drop and reverse cymbals.
- **Master:** a 25 Hz high-pass, 2:1 RMS glue compression, a lookahead peak limiter, a tanh soft clip and true-peak normalisation.

## Arrangement (bars)

| Bars | Section | What's playing |
|---|---|---|
| 1–16 | DJ intro | kick, rims, closed hats from bar 9 |
| 17–32 | build | + rumble, clap, shaker, open hats |
| 33–48 | groove | + rolling bass |
| 49–64 | arp enters | filtered arp, dub stabs from bar 57, kick drop-out on bar 64 |
| 65–96 | main groove | arp cutoff opens, ride, reverse cymbal, 2-beat kick gap |
| 97–120 | breakdown | no kick or bass; pad, blooming arp, 8-bar riser and snare roll |
| 121–160 | drop | everything in, crash and sub-drop; arp variation at bars 137–152 |
| 161–176 | strip-down | arp filter closes, arp and stabs exit at bar 169 |
| 177–192 | DJ outro | kick, rumble, hats, perc; elements thin out for mixing |

## Regenerate / tweak

```bash
pip install numpy scipy        # ffmpeg is needed for the mp3
python3 techno/generate.py             # full render, about 3 min
python3 techno/generate.py --mix-only  # re-balance from cached stems, about 30 s
```

Mix levels are in `MIX`, a target dB relative to the kick for each stem. Section layouts are the `*_BARS` sets. The arp melody is `ARP_A`/`ARP_B`, harmony is `CHORDS`, and cutoff automation is the `curve([...])` keyframes in `render_arp`.
