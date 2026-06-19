#!/usr/bin/env python3
"""
Face Recognition Pipeline
=========================
End-to-end: detect faces → build database → live recognition.

Usage:
  python pipeline.py collect  ./raw_images  --output data
  python pipeline.py build-db --data data/train
  python pipeline.py live     --db database.pkl
  python pipeline.py math     --db database.pkl
"""

import subprocess
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Pipeline")
    parser.add_argument("command", choices=[
        "collect", "build-db", "live", "math",
    ], help="Pipeline step")
    parser.add_argument("input", nargs="?", help="Input path")
    parser.add_argument("--data", "-d", default="data/train", help="Training data directory")
    parser.add_argument("--output", "-o", default="data", help="Output directory (for collect)")
    parser.add_argument("--db", default="database.pkl", help="Database file")
    parser.add_argument("--threshold", "-t", type=float, default=0.60, help="Combined similarity threshold")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device ID")
    parser.add_argument("--no-display", action="store_true", help="Run without GUI")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    if args.command == "collect":
        if not args.input:
            print("Usage: python pipeline.py collect <raw_dir>")
            sys.exit(1)
        subprocess.run([
            sys.executable, str(script_dir / "face_recognition" / "collect.py"),
            args.input, "--output", args.output,
            "--val-split", str(args.val_split),
            "--test-split", str(args.test_split),
        ])

    elif args.command == "build-db":
        subprocess.run([
            sys.executable, str(script_dir / "face_recognition" / "recognizer.py"),
            "build-db", "--data", args.data, "--output", args.db,
        ])

    elif args.command == "live":
        cmd = [
            sys.executable, str(script_dir / "face_recognition" / "recognizer.py"),
            "live", "--db", args.db,
            "--threshold", str(args.threshold),
            "--camera", str(args.camera),
        ]
        if args.no_display:
            cmd.append("--no-display")
        subprocess.run(cmd)

    elif args.command == "math":
        subprocess.run([
            sys.executable, str(script_dir / "face_recognition" / "recognizer.py"),
            "math", "--db", args.db,
        ])


if __name__ == "__main__":
    main()
