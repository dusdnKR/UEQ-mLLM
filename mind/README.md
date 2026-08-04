# MIND — Multimodal INteractive Dashboard

MIND is the web-based front end that captures the user signals UEQ-mLLM turns into questionnaires. It was introduced in our ICCE-Asia 2023 paper and extended for the 2024 work described in the [top-level README](../README.md).

<video src="https://raw.githubusercontent.com/dusdnKR/UEQ-mLLM/main/assets/mind-demo.mp4" controls width="100%"></video>

## Modules

| Module | What it does | Endpoint |
|--------|--------------|----------|
| Emotion recognition | DeepFace over the webcam feed, pretrained on FER-2013, classifies seven emotions: angry, disgust, fear, happy, sad, surprise, neutral | `/emotion_feed` |
| Attention recognition | Emotiv Epoc EEG over Lab Streaming Layer; a Blackman-windowed short-time Fourier transform feeds a KNN classifier for focus / unfocus / drowsy | `/attention_feed` |
| EEG stream | Raw 14-channel traces, streamed as server-sent events | `/eeg_feed` |
| EEG topography | MNE scalp topomaps rendered per sample | `/mne_feed` |
| Affective image generation | Prompted by the detected attention and emotion state | `/diffusion_feed`, `/streamdiffusion_feed` |
| Face detection | Haar cascade overlay on the webcam feed | `/face_feed` |

The 2024 paper replaces the ControlNet-based image generator with [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion), which cuts computational cost enough to generate at over 10 frames per second. `app.py` retains the ControlNet path; `app_loaded.py` adds the StreamDiffusion endpoint.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Flask backend using live EEG through the Emotiv LSL stream |
| `app_loaded.py` | Flask backend using prerecorded EEG from `datas/`, plus StreamDiffusion |
| `dash.py` | Streamlit front end that arranges the streaming endpoints into the dashboard |
| `stream_diffusion_demo.py` | Standalone webcam-to-image StreamDiffusion demo |
| `templates/` | One HTML page per streaming endpoint |
| `utils/wrapper.py` | StreamDiffusion wrapper, adapted from the upstream project |

## Setup

1. Download the `models/` folder from [Google Drive](https://drive.google.com/drive/folders/1djwUiAWDnatcuyIDgYtblJyOTAx_YTBW?usp=sharing) and merge it into `mind/models/`. It contains `saved_model*` (KNN attention classifiers) and `scaler_knn*.joblib`, which are too large to version here. `haarcascade_frontalface_alt.xml` is already included.
2. Install from `requirements.txt`. Several pins conflict with one another, so installing the packages you actually import tends to work better than installing the file wholesale. `requirements-streamdiffusion.txt` covers the separate StreamDiffusion environment.
3. Run the backend from **this** directory — the modules resolve `models/`, `datas/`, and `utils/` relative to the working directory:
   - live EEG (requires the Emotiv Lab Streaming Layer): `cd mind && python app.py`
   - prerecorded EEG: `cd mind && python app_loaded.py`
4. In a second shell, also from `mind/`: `streamlit run dash.py`

Flask serves on port 5000 and Streamlit polls those endpoints, so both processes need to be running.

Recorded EEG files are not included in this repository. `app_loaded.py` reads `datas/eeg_record3.mat` with 14 channels at 128 Hz, laid out on the standard 10-20 montage.
