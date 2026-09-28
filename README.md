# Drum Rebuilder — Mobile Cloud MVP

A phone-friendly Streamlit app that records or uploads music and rebuilds a clean **kick / snare / hi-hat** performance as WAV + MIDI.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository (for example `drum-rebuilder`).
2. Upload all files in this folder, including `.streamlit/config.toml`.
3. Go to https://share.streamlit.io and create an app from the repository.
4. Entrypoint: `app.py`.
5. In Advanced settings choose Python **3.12**.
6. Deploy. Streamlit will provide a `*.streamlit.app` URL.
7. Open that URL on Android/iPhone and allow microphone access.

`requirements.txt` contains Python dependencies. `packages.txt` installs FFmpeg and libsndfile on Streamlit's Linux host.

## Phone workflow

1. Open the app URL on your phone.
2. Choose **Record**.
3. Tap the microphone and allow browser microphone access.
4. Play 20–60 seconds of the source song through another speaker/device.
5. Stop recording and tap **Analyze & rebuild drums**.
6. Listen to the rebuilt drums or download WAV/MIDI.

The server analyzes at most the first 90 seconds to keep the MVP responsive.

## Current model

This version deliberately validates the product workflow before introducing a heavyweight neural transcription model. It detects onsets and classifies three instruments using spectral features:

- Kick — General MIDI 36
- Snare — General MIDI 38
- Closed hi-hat — General MIDI 42

Timing and velocity are retained. Hits are not automatically quantized.

## Next accuracy upgrade

Replace the heuristic classifier with a learned automatic drum transcription model, then add toms, crash, ride, open/pedal hi-hat, ghost notes and per-instrument editing. A production deployment will likely need more compute than Streamlit Community Cloud for heavier separation/transcription models.

## Local run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
