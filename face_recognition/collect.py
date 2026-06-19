import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import json
import random
import argparse


CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def detect_faces(image_path: str, min_face_size: int = 80) -> list:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [!] Could not read: {image_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_size, min_face_size)
    )

    # Filter to largest face (assume one person per image for training)
    if len(faces) == 0:
        return []

    largest = max(faces, key=lambda f: f[2] * f[3])
    return [largest]


def extract_face_crop(img: np.ndarray, face: tuple, output_size: tuple = (160, 160)) -> np.ndarray:
    x, y, w, h = face
    padding = int(0.2 * max(w, h))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + h + padding)

    face_crop = img[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, output_size, interpolation=cv2.INTER_LANCZOS4)
    return face_crop


def organize_dataset(raw_dir: str, output_dir: str, val_split: float = 0.2, test_split: float = 0.1):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    if not raw_path.exists():
        print(f"Raw directory not found: {raw_dir}")
        return

    for split in ["train", "val", "test"]:
        (out_path / split).mkdir(parents=True, exist_ok=True)

    skipped = []
    face_counts = {}

    person_dirs = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    if not person_dirs:
        print(f"No person subdirectories found in {raw_dir}")
        print("Expected structure: raw_dir/person_name/*.jpg")
        return

    for person_dir in tqdm(person_dirs, desc="Processing people"):
        person_name = person_dir.name
        image_paths = sorted([
            p for p in person_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ])

        if not image_paths:
            print(f"  [!] No images for {person_name}")
            continue

        face_paths = []
        for img_path in image_paths:
            faces = detect_faces(str(img_path))
            if not faces:
                skipped.append(str(img_path))
                continue

            img = cv2.imread(str(img_path))
            face_crop = extract_face_crop(img, faces[0])

            face_dir = out_path / "train" / person_name
            face_dir.mkdir(parents=True, exist_ok=True)

            face_filename = f"{img_path.stem}_face{img_path.suffix}"
            cv2.imwrite(str(face_dir / face_filename), face_crop)
            face_paths.append(str(face_dir / face_filename))

        face_counts[person_name] = len(face_paths)

        # Split into train/val/test
        n = len(face_paths)
        if n >= 5:
            n_val = max(1, int(n * val_split))
            n_test = max(1, int(n * test_split))
            remaining = n - n_val - n_test
            if remaining < 1:
                n_val = max(1, n // 5)
                n_test = max(1, n // 5)
                remaining = n - n_val - n_test

            indices = list(range(n))
            random.Random(42).shuffle(indices)
            test_idx = set(indices[:n_test])
            val_idx = set(indices[n_test:n_test + n_val])

            for i, path_str in enumerate(face_paths):
                p = Path(path_str)
                if i in test_idx:
                    dest = out_path / "test" / person_name / p.name
                elif i in val_idx:
                    dest = out_path / "val" / person_name / p.name
                else:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest))

    # Print summary
    print("\n=== Collection Summary ===")
    for person, count in face_counts.items():
        print(f"  {person}: {count} face crops")
    print(f"  Skipped (no face detected): {len(skipped)} images")
    print(f"  Output: {out_path}")

    # Save metadata
    meta = {
        "people": face_counts,
        "skipped": skipped,
        "total_people": len(face_counts),
        "total_faces": sum(face_counts.values()),
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nMetadata saved to {out_path / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and organize face data")
    parser.add_argument("raw_dir", help="Directory with person subfolders containing images")
    parser.add_argument("--output", "-o", default="data", help="Output directory (default: data)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio")
    args = parser.parse_args()

    organize_dataset(args.raw_dir, args.output, args.val_split, args.test_split)
