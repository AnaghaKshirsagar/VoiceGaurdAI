from transformers import pipeline

# Lightweight Whisper model for deployment
MODEL_NAME = "openai/whisper-tiny"

print("Loading VoiceGuard AI speech-to-text model...")

transcriber = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    device=-1,
)

print("Speech-to-text model loaded successfully!")


def transcribe_audio(file_path):
    """
    Transcribe an audio file using Whisper Tiny.

    Returns:
        str: Full transcript
    """

    print("\nStarting audio transcription...")

    try:
        result = transcriber(
            file_path,
            chunk_length_s=30,
            stride_length_s=(5, 5),
            return_timestamps=True,
            generate_kwargs={
                "task": "transcribe",
                "language": "english",
            },
        )

        print("\nWhisper result:")
        print(result)

        chunks = result.get("chunks", [])

        transcript_parts = []

        for chunk in chunks:
            text = chunk.get("text", "").strip()

            if text:
                transcript_parts.append(text)

        if transcript_parts:
            transcript = " ".join(transcript_parts)
        else:
            transcript = result.get("text", "").strip()

        transcript = transcript.strip()

        print("\n" + "=" * 60)
        print("TRANSCRIPT")
        print("=" * 60)
        print(transcript)
        print("=" * 60)

        return transcript

    except Exception as e:
        print(f"Transcription error: {e}")

        # Don't crash the entire VoiceGuard analysis if transcription fails.
        return ""