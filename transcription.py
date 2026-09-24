from transformers import pipeline

MODEL_NAME = "openai/whisper-small"

print("Loading VoiceGuard AI speech-to-text model...")

transcriber = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    device=-1
)

print("Speech-to-text model loaded successfully!")


def transcribe_audio(file_path):
    print("\nStarting FULL AUDIO transcription...")

    result = transcriber(
        file_path,
        chunk_length_s=30,
        stride_length_s=(5, 5),
        return_timestamps=True,
        generate_kwargs={
            "task": "transcribe",
            "language": "english",
            "condition_on_prev_tokens": True,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05
        }
    )

    print("\nWhisper raw result:")
    print(result)

    # Get all Whisper segments
    chunks = result.get("chunks", [])

    transcript_parts = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        if text:
            transcript_parts.append(text)

    # If Whisper did not return chunks, use normal text output
    if transcript_parts:
        transcript = " ".join(transcript_parts)
    else:
        transcript = result.get("text", "").strip()

    transcript = transcript.strip()

    print("\n" + "=" * 60)
    print("FULL TRANSCRIPT")
    print("=" * 60)
    print(transcript)
    print("=" * 60)

    return transcript