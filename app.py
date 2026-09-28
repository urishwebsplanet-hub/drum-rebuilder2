from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path

import librosa
import mido
import numpy as np
import soundfile as sf
import streamlit as st
from streamlit.components.v1 import declare_component

music_recorder = declare_component("music_recorder", path=str(Path(__file__).parent / "music_recorder"))

APP_TITLE = "Drum Rebuilder"
SR = 44100
HOP = 256
MAX_SECONDS = 90

MIDI_NOTES = {"kick": 36, "snare": 38, "hihat": 42}


@st.cache_data(show_spinner=False)
def read_audio_bytes(data: bytes, suffix: str):
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    path = Path(name)
    try:
        path.write_bytes(data)
        y, sr = librosa.load(path, sr=SR, mono=True, duration=MAX_SECONDS)
        return y.astype(np.float32), sr
    finally:
        path.unlink(missing_ok=True)


def _band_energy(mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.mean(mag[mask] ** 2))


def transcribe_three_piece(y: np.ndarray, sr: int):
    if y.size == 0:
        return [], 120.0
    y = librosa.util.normalize(y)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP,
        backtrack=False,
        units="frames",
        pre_max=3,
        post_max=3,
        pre_avg=5,
        post_avg=5,
        delta=0.12,
        wait=2,
    )
    if len(onset_frames) == 0:
        return [], 120.0

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP, win_length=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    strengths = onset_env[np.clip(onset_frames, 0, len(onset_env) - 1)]
    s95 = max(float(np.percentile(strengths, 95)), 1e-8)

    events = []
    for frame, strength in zip(onset_frames, strengths):
        frame = int(min(frame, S.shape[1] - 1))
        mag = S[:, frame]
        low = _band_energy(mag, freqs, 35, 180)
        mid = _band_energy(mag, freqs, 180, 2500)
        high = _band_energy(mag, freqs, 4500, 16000)
        total = low + mid + high + 1e-12
        low_r, mid_r, high_r = low / total, mid / total, high / total
        centroid = float(librosa.feature.spectral_centroid(S=mag[:, None], sr=sr)[0, 0])

        if low_r > 0.50 and centroid < 1800:
            inst = "kick"
            confidence = min(1.0, 0.55 + low_r)
        elif high_r > 0.46 or centroid > 5200:
            inst = "hihat"
            confidence = min(1.0, 0.45 + high_r)
        else:
            inst = "snare"
            confidence = min(1.0, 0.45 + mid_r)

        velocity = int(np.clip(35 + 92 * (float(strength) / s95), 25, 127))
        events.append(
            {
                "time": round(float(librosa.frames_to_time(frame, sr=sr, hop_length=HOP)), 3),
                "instrument": inst,
                "velocity": velocity,
                "confidence": round(confidence, 3),
            }
        )

    tempo_arr = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    tempo = float(tempo_arr[0]) if len(tempo_arr) else 120.0
    return events, tempo


def midi_bytes(events, tempo: float) -> bytes:
    mid = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    tempo = max(30.0, min(300.0, tempo))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    ticks_per_second = mid.ticks_per_beat * 1_000_000 / mido.bpm2tempo(tempo)
    last_tick = 0
    for ev in sorted(events, key=lambda x: x["time"]):
        tick = int(round(ev["time"] * ticks_per_second))
        delta = max(0, tick - last_tick)
        note = MIDI_NOTES[ev["instrument"]]
        track.append(mido.Message("note_on", channel=9, note=note, velocity=ev["velocity"], time=delta))
        track.append(mido.Message("note_off", channel=9, note=note, velocity=0, time=24))
        last_tick = tick + 24
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def _noise_burst(length: int, decay: float, rng: np.random.Generator):
    n = rng.standard_normal(length)
    env = np.exp(-np.linspace(0, decay, length))
    return n * env


def render_drums(events, duration: float, sr: int = SR) -> bytes:
    n = int((duration + 1.0) * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(7)
    for ev in events:
        start = int(ev["time"] * sr)
        amp = ev["velocity"] / 127.0
        inst = ev["instrument"]
        if inst == "kick":
            length = int(0.34 * sr)
            t = np.arange(length) / sr
            freq = 105 * np.exp(-t * 18) + 42
            phase = 2 * np.pi * np.cumsum(freq) / sr
            hit = np.sin(phase) * np.exp(-t * 12)
        elif inst == "snare":
            length = int(0.24 * sr)
            t = np.arange(length) / sr
            hit = 0.82 * _noise_burst(length, 13, rng) + 0.28 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 18)
        else:
            length = int(0.10 * sr)
            t = np.arange(length) / sr
            noise = _noise_burst(length, 28, rng)
            hit = np.concatenate([[noise[0]], np.diff(noise)]) * np.exp(-t * 18)
        end = min(start + len(hit), n)
        if start < n:
            out[start:end] += (amp * hit[: end - start]).astype(np.float32)
    peak = np.max(np.abs(out)) if len(out) else 0
    if peak > 0:
        out = 0.92 * out / peak
    buf = io.BytesIO()
    sf.write(buf, out, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def mobile_css():
    st.markdown(
        """
        <style>
        .block-container {max-width: 760px; padding-top: 1.1rem; padding-bottom: 2rem;}
        div.stButton > button {width: 100%; min-height: 3.2rem; font-size: 1.08rem; font-weight: 700;}
        div.stDownloadButton > button {width: 100%; min-height: 2.8rem;}
        [data-testid="stFileUploader"] {padding: .35rem 0;}
        [data-testid="stMetricValue"] {font-size: 1.55rem;}
        h1 {font-size: 2rem !important;}
        @media (max-width: 640px) {
          .block-container {padding-left: 1rem; padding-right: 1rem; padding-top: .75rem;}
          h1 {font-size: 1.75rem !important;}
          [data-testid="stHorizontalBlock"] {gap: .45rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🥁", layout="centered")
    mobile_css()
    st.title("🥁 Drum Rebuilder")
    st.caption("Record or upload music → rebuild kick, snare and hi-hat → listen or export MIDI")

    source = st.segmented_control("Input", ["Record", "Upload"], default="Record")
    audio = None
    suffix = ".wav"

    if source == "Record":
        st.write("Tap the microphone, then play the song through another speaker/device.")
        st.caption("Music recorder v2 — speech processing off when supported. Keep this page visible while recording.")
        recording = music_recorder(key="music_capture", default=None)
        if recording and recording.get("wav"):
            try:
                audio = io.BytesIO(base64.b64decode(recording["wav"], validate=True))
            except (ValueError, TypeError):
                st.error("Recording could not be read. Please record again.")
            with st.expander("Microphone settings"):
                st.json({k: recording.get("settings", {}).get(k, "Not reported") for k in ("echoCancellation", "noiseSuppression", "autoGainControl", "sampleRate")})
    else:
        audio = st.file_uploader("Choose audio", type=["wav", "mp3", "m4a", "flac", "ogg"])
        if audio is not None:
            suffix = Path(audio.name).suffix or ".wav"

    if audio is None:
        st.info("For the first test, use 20–60 seconds with a clear drum groove.")
        st.stop()

    st.audio(audio)
    st.download_button("Save original recording", audio.getvalue(), "original" + suffix, "audio/wav" if suffix == ".wav" else "application/octet-stream")
    if not st.button("Analyze & rebuild drums", type="primary"):
        st.stop()

    raw = audio.getvalue()
    with st.spinner("Listening for drum hits…"):
        try:
            y, sr = read_audio_bytes(raw, suffix)
            events, tempo = transcribe_three_piece(y, sr)
            midi = midi_bytes(events, tempo)
            rendered = render_drums(events, len(y) / sr, sr)
        except Exception as exc:
            st.error(f"Could not process this recording: {exc}")
            st.stop()

    if not events:
        st.warning("I couldn't detect clear drum hits in this clip. Try a louder/cleaner section.")
        st.stop()

    counts = {k: sum(e["instrument"] == k for e in events) for k in MIDI_NOTES}
    st.success("Reconstruction complete")
    st.metric("Estimated tempo", f"{tempo:.1f} BPM")
    c1, c2, c3 = st.columns(3)
    c1.metric("Kick", counts["kick"])
    c2.metric("Snare", counts["snare"])
    c3.metric("Hi-hat", counts["hihat"])

    st.subheader("Rebuilt drums")
    st.audio(rendered, format="audio/wav")
    st.download_button("Download drum WAV", rendered, "reconstructed_drums.wav", "audio/wav")
    st.download_button("Download MIDI", midi, "drums.mid", "audio/midi")

    with st.expander("Detected hits"):
        st.dataframe(events, use_container_width=True, hide_index=True)

    st.caption("MVP: timing is preserved; hits are not quantized. Current model recognizes kick, snare and closed hi-hat only.")


if __name__ == "__main__":
    main()
