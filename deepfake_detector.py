import numpy as np
import librosa
import onnxruntime as ort
from huggingface_hub import hf_hub_download


# ============================================================
# MODEL
# ============================================================

MODEL_REPO = "ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model"
MODEL_FILENAME = "best_model.onnx"

print("Loading VoiceGuard AI deepfake detector...")

MODEL_PATH = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILENAME
)

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("Dhwani deepfake detector loaded successfully!")


# ============================================================
# AUDIO SETTINGS
# ============================================================

TARGET_SAMPLE_RATE = 16000

CHUNK_SECONDS = 3.0
CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_SECONDS)

HOP_SECONDS = 1.5
HOP_SAMPLES = int(TARGET_SAMPLE_RATE * HOP_SECONDS)


# ============================================================
# MODEL INFERENCE
# ============================================================

def classify_chunk(audio):
    """
    Classify one 3-second audio chunk.

    Returns:
        fake_probability
        real_probability
    """

    audio = np.asarray(audio, dtype=np.float32)

    # Normalize
    mean = np.mean(audio)
    variance = np.var(audio)

    audio = (audio - mean) / np.sqrt(
        variance + 1e-5
    )

    # Make exactly 48,000 samples
    if len(audio) > CHUNK_SAMPLES:

        audio = audio[:CHUNK_SAMPLES]

    elif len(audio) < CHUNK_SAMPLES:

        padding = CHUNK_SAMPLES - len(audio)

        audio = np.pad(
            audio,
            (0, padding),
            mode="constant"
        )

    audio = audio.astype(
        np.float32
    ).reshape(
        1,
        CHUNK_SAMPLES
    )

    # Run ONNX model
    outputs = session.run(
        None,
        {
            input_name: audio
        }
    )

    logits = outputs[0]

    # Softmax
    logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True
    )

    probabilities = (
        np.exp(logits)
        /
        np.sum(
            np.exp(logits),
            axis=1,
            keepdims=True
        )
    )

    # Dhwani model:
    # class 0 = real
    # class 1 = fake

    real_probability = float(
        probabilities[0][0]
    )

    fake_probability = float(
        probabilities[0][1]
    )

    return (
        fake_probability,
        real_probability
    )


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_deepfake(file_path):

    print("\n" + "=" * 60)
    print("VOICEGUARD AI - DEEPFAKE AUDIO ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, sample_rate = librosa.load(
        file_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True
    )

    duration = len(audio) / TARGET_SAMPLE_RATE

    print(
        f"Audio duration: {duration:.2f} seconds"
    )

    # --------------------------------------------------------
    # Remove silence
    # --------------------------------------------------------

    try:

        trimmed_audio, _ = librosa.effects.trim(
            audio,
            top_db=30
        )

    except Exception:

        trimmed_audio = audio

    if len(trimmed_audio) < TARGET_SAMPLE_RATE:

        trimmed_audio = audio

    # --------------------------------------------------------
    # Create overlapping chunks
    # --------------------------------------------------------

    chunks = []

    start = 0

    while start < len(trimmed_audio):

        end = start + CHUNK_SAMPLES

        chunk = trimmed_audio[start:end]

        # Ignore extremely short chunks
        if len(chunk) >= int(
            TARGET_SAMPLE_RATE * 0.75
        ):

            chunks.append(chunk)

        if end >= len(trimmed_audio):
            break

        start += HOP_SAMPLES

    # If nothing was produced
    if not chunks:

        chunks = [trimmed_audio]

    print(
        f"Chunks analyzed: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Analyze each chunk
    # --------------------------------------------------------

    fake_scores = []
    real_scores = []

    chunk_results = []

    for index, chunk in enumerate(chunks):

        fake_score, real_score = classify_chunk(
            chunk
        )

        fake_scores.append(
            fake_score
        )

        real_scores.append(
            real_score
        )

        chunk_result = {
            "chunk": index + 1,
            "fake_score": fake_score,
            "real_score": real_score
        }

        chunk_results.append(
            chunk_result
        )

        print(
            f"Chunk {index + 1}: "
            f"FAKE={fake_score * 100:.2f}% "
            f"REAL={real_score * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Aggregate results
    # --------------------------------------------------------

    average_fake = float(
        np.mean(fake_scores)
    )

    median_fake = float(
        np.median(fake_scores)
    )

    average_real = float(
        np.mean(real_scores)
    )

    # Percentage of chunks with strong fake signal
    strong_fake_ratio = float(
        np.mean(
            np.array(fake_scores) >= 0.70
        )
    )

    # Percentage of chunks with at least
    # moderate fake signal
    fake_chunk_ratio = float(
        np.mean(
            np.array(fake_scores) >= 0.50
        )
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    final_fake_score = (
        average_fake * 0.50
        +
        median_fake * 0.30
        +
        strong_fake_ratio * 0.20
    )

    final_fake_score = float(
        np.clip(
            final_fake_score,
            0.0,
            1.0
        )
    )

    final_real_score = float(
        1.0 - final_fake_score
    )

    # --------------------------------------------------------
    # Assessment
    # --------------------------------------------------------

    if (
        final_fake_score >= 0.75
        and strong_fake_ratio >= 0.50
    ):

        assessment = (
            "Potential Synthetic Audio"
        )

    elif (
        final_fake_score >= 0.50
        or fake_chunk_ratio >= 0.50
    ):

        assessment = (
            "Suspicious Synthetic Signal"
        )

    elif final_real_score >= 0.70:

        assessment = (
            "Potential Authentic Audio"
        )

    else:

        assessment = (
            "Uncertain Audio"
        )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    distance_from_uncertain = abs(
        final_fake_score - 0.50
    )

    confidence = min(
        1.0,
        distance_from_uncertain * 2
    )

    # --------------------------------------------------------
    # Print final result
    # --------------------------------------------------------

    print("\n" + "-" * 60)

    print(
        f"Average fake signal: "
        f"{average_fake * 100:.2f}%"
    )

    print(
        f"Median fake signal: "
        f"{median_fake * 100:.2f}%"
    )

    print(
        f"Strong fake chunks: "
        f"{strong_fake_ratio * 100:.2f}%"
    )

    print(
        f"Final synthetic score: "
        f"{final_fake_score * 100:.2f}%"
    )

    print(
        f"Assessment: {assessment}"
    )

    print("-" * 60)

    return {

        "assessment": assessment,

        "fake_score": final_fake_score,

        "real_score": final_real_score,

        "confidence": confidence,

        "average_fake_score": average_fake,

        "median_fake_score": median_fake,

        "fake_chunk_ratio": fake_chunk_ratio,

        "strong_fake_ratio": strong_fake_ratio,

        "chunks_analyzed": len(chunks),

        "duration": duration,

        "raw_results": chunk_results
    }