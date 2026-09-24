import hashlib
import librosa

from transcription import transcribe_audio
from risk_engine import analyze_fraud_signals
from threat_assessment import calculate_threat_assessment
from deepfake_detector import detect_deepfake


def calculate_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            data = file.read(8192)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


def analyze_audio(file_path):

    # Load uploaded audio
    audio, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    duration = librosa.get_duration(
        y=audio,
        sr=sample_rate
    )

    # Generate file hash
    file_hash = calculate_hash(file_path)

    # Convert speech to text
    transcript = transcribe_audio(file_path)

    # Analyze fraud language
    fraud_result = analyze_fraud_signals(
        transcript
    )

    # Detect AI-generated / synthetic voice
    deepfake_result = detect_deepfake(
        file_path
    )

    # Calculate overall threat
    threat_result = calculate_threat_assessment(
        fake_score=deepfake_result["fake_score"],
        fraud_risk_level=fraud_result["risk_level"],
        fraud_points=fraud_result["risk_points"]
    )

    return {
        "audio": audio,
        "sample_rate": sample_rate,
        "duration": duration,
        "hash": file_hash,
        "transcript": transcript,

        "fraud_analysis": fraud_result,

        "deepfake_analysis": deepfake_result,

        "threat_assessment": threat_result
    }