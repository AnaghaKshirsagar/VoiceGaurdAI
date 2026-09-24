# VoiceguardAI
### AI-Powered Voice Fraud & Deepfake Detection System

Voice Guard AI is an AI-powered audio forensics platform designed to analyze
voice recordings for potential synthetic voice, deepfake, and fraud-related
indicators.

The system combines audio analysis, voice transcription, synthetic voice
detection, fraud signal analysis, threat assessment, and digital evidence
integrity into a single interactive dashboard.


## 🖥️ Application Preview
## Screenshots
### Main Dashboard
<img width="1600" height="803" alt="frontpage" src="https://github.com/user-attachments/assets/fd7d56d2-02cf-43da-826c-0f14bc741137">

### Voice Analysis Dashboard
<img width="1409" height="898" alt="voice analysis" src="https://github.com/user-attachments/assets/2319feb3-bab3-4713-9dfb-7cf439274a1d">



## Risk Classification
<img width="2870" height="1382" alt="voiceguard_high_high_adjacent_2x2" src="https://github.com/user-attachments/assets/329e4157-d5ce-475a-a4f0-e156d60bc52a">



## Features
- 🎙️ Audio file analysis
- 🤖 Synthetic/deepfake voice detection
- 📝 Automatic voice transcription
- 🚨 Fraud signal analysis
- 🧠 Threat assessment
- 📊 Audio waveform visualization
- 🌈 Spectrogram analysis
- 🔐 SHA-256 file integrity hashing
- 📁 Evidence-oriented analysis
- 🖥️ Interactive Streamlit dashboard

## 📁Project Structure

```text
Voice-Guard-AI/
│
├── app.py
│   └── Streamlit user interface
│
├── audio_analysis.py
│   └── Main audio analysis pipeline
│
├── transcription.py
│   └── Speech-to-text processing
│
├── detector.py
│   └── Synthetic/deepfake voice detection
│
├── risk_engine.py
│   └── Fraud signal analysis
│
├── threat_assessment.py
│   └── Threat assessment logic
│
├── requirements.txt
│   └── Python dependencies
│
├── uploads/
│   └── Uploaded audio files
│
├── screenshots/
│   └── Application screenshots
│
└── README.md
    └── Project documentation
```
## 🧠 How Voice Guard AI Works

```text
              ┌───────────────────┐
              │    Audio Upload   │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │  Audio Analysis   │
              └─────────┬─────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌───────────┐  ┌─────────────┐
    │Transcrip-│  │ Deepfake  │  │   Audio     │
    │  tion    │  │ Detection │  │  Forensics  │
    └────┬─────┘  └─────┬─────┘  └──────┬──────┘
         │              │               │
         └──────────────┼───────────────┘
                        ▼
              ┌───────────────────┐
              │  Fraud Analysis   │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ Threat Assessment │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │ Forensic Results  │
              └───────────────────┘
```


## ⚙️Installation

### Clone the repository

```text git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Voice-Guard-AI
```

## Create a virtual environment
```text
python -m venv venv
```

## Activate the environment
### Windows
```text
venv\Scripts\activate
```
## Install dependencies
```text
pip install -r requirements.txt
```
## Running the Application
### Start the Streamlit application:

```text
streamlit run app.py
```

## 👥Team Name : APEXCODERS

Team Members:


1) Anagha Kshirsagar

2) Ritika Kumawat

3) Riya Kumawat

4) Samruddhi Kumavat

5) Samrudhi Moon

6) Aditi Ippar


## 🔮 Future Scope

1.Real-time voice analysis

2.Investigator case management

3.Advanced deepfake detection models

4.Speaker verification

5.Audio manipulation detection



 



