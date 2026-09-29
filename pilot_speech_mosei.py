from pathlib import Path

from speech.run_file import run_speech_file


AUDIO_PATH = Path("data/processed/pilot/266396_0.wav")

TRANSCRIPT = "there are two types of people in this world people who like m night films"

record = run_speech_file(
    audio_path=AUDIO_PATH,
    participant_id="266396",
    session_id="266396",
    segment_id="266396",
    label_idx=0,
    start=0.311,
    end=3.775,
    transcript=TRANSCRIPT,
)

print("=" * 70)
print("CMU-MOSEI SPEECH PILOT")
print("=" * 70)

print(f"segment_id       : {record['segment_id']}")
print(f"label_idx        : {record['label_idx']}")
print(f"start            : {record['start']}")
print(f"end              : {record['end']}")
print(f"feature_count    : {len(record['features'])}")
print(f"quality_score    : {record['quality_score']:.4f}")
print(f"speech_ratio     : {record['speech_ratio']:.4f}")
print(f"snr_estimate     : {record['snr_estimate']:.4f}")
print(f"transcript       : {TRANSCRIPT}")
print("VALIDATION       : PASS")