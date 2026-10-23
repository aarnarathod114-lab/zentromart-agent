"""Record a few sample customer voice messages into sample_audio/ (your own voice only)."""
from pathlib import Path

import soundfile as sf

from voice import SR, record

PROMPTS = [
    ("q1_return_window", "What is the return window for electronics?"),
    ("q2_refund_upi", "If my refund goes to UPI, how many days will it take?"),
    ("q3_hinglish_cod", "Kya cash on delivery pe extra charge lagta hai?"),
    ("q4_unanswerable", "Can I schedule a delivery at midnight?"),
    ("q5_unclear", "(stay silent, or just mumble - tests unclear audio)"),
]

out = Path("sample_audio")
out.mkdir(exist_ok=True)
for name, text in PROMPTS:
    input(f"\nNext: {text}\nPress Enter, then speak... ")
    audio = record(6)
    sf.write(out / f"{name}.wav", audio, SR)
    print(f"   saved sample_audio/{name}.wav (peak level {int(abs(audio).max())})")
print("\nDone: 5 files in sample_audio/")