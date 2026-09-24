from deepfake_detector import detect_deepfake


AUDIO_FILE = "test_audio.wav"


result = detect_deepfake(
    AUDIO_FILE
)


print("\n")
print("=" * 60)
print("VOICEGUARD AI TEST RESULT")
print("=" * 60)

print(
    f"Assessment: "
    f"{result['assessment']}"
)

print(
    f"Synthetic signal: "
    f"{result['fake_score'] * 100:.2f}%"
)

print(
    f"Authentic signal: "
    f"{result['real_score'] * 100:.2f}%"
)

print(
    f"Confidence: "
    f"{result['confidence'] * 100:.2f}%"
)

print(
    f"Chunks analyzed: "
    f"{result['chunks_analyzed']}"
)

print("=" * 60)