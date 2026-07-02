"""
Enhance database.pkl with student profile data.
Run: python enhance_db.py

This adds roll_number, gmail, program, semester, section, etc.
to each person entry in the face recognition database.
"""
import pickle
from pathlib import Path

DB_PATH = Path(__file__).parent / "database.pkl"

STUDENT_PROFILES = {
    "shahzaib": {
        "name": "Muhammad Shahzeb",
        "roll_number": "SP22-BSE-073",
        "gmail": "razashahzaib119@gmail.com",
        "program": "BSE (Computer Science)",
        "semester": 6,
        "section": "A",
        "phone": "+92-300-1111111",
        "father_name": "Ahmed Ali",
        "address": "Lahore, Pakistan",
    },
    "ahsan": {
        "name": "Muhammad Ahsan",
        "roll_number": "FA22-BS-073",
        "gmail": "ahsan@gmail.com",
        "program": "BSE (Computer Science)",
        "semester": 4,
        "section": "B",
        "phone": "+92-300-2222222",
        "father_name": "Ahsan Father",
        "address": "Islamabad, Pakistan",
    },
    "arslan": {
        "name": "Arslan Rathore",
        "roll_number": "FA22-BCS-045",
        "gmail": "arslan@gmail.com",
        "program": "BSCS",
        "semester": 4,
        "section": "B",
        "phone": "+92-300-2222222",
        "father_name": "Rathore",
        "address": "Islamabad, Pakistan",
    },
    "faraz": {
        "name": "Faraz Khan",
        "roll_number": "FA22-BAI-012",
        "gmail": "faraz@gmail.com",
        "program": "BSAI",
        "semester": 4,
        "section": "A",
        "phone": "+92-300-3333333",
        "father_name": "Khan",
        "address": "Karachi, Pakistan",
    },
}


def enhance():
    if not DB_PATH.exists():
        print(f"[!] database.pkl not found at {DB_PATH}")
        return

    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)

    for person_key in list(db["people"].keys()):
        profile = STUDENT_PROFILES.get(person_key)
        if profile:
            db["people"][person_key].update(profile)
            print(f"  [+] Added profile for {person_key} ({profile['name']})")
        else:
            print(f"  [-] No profile defined for {person_key}")

    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

    print(f"\n  Database enhanced with student profiles.")
    print(f"  Total people: {len(db['people'])}")
    for k, v in db["people"].items():
        print(f"    {k}: {v.get('name', '?')} | {v.get('roll_number', '?')} | {v.get('gmail', '?')}")


if __name__ == "__main__":
    enhance()
