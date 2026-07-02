"""
Face Recognition API — FastAPI server.

Start:  python api.py
       (runs on http://localhost:8000)

Endpoints:
  POST /recognize    — upload an image, get back identified student profile
  GET  /health       — health check
"""
import os
import pickle
import io
import uvicorn
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from face_recognition import load_model, get_embedding, load_database, recognize

SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / "database.pkl"

app = FastAPI(title="SafeCampus Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
db = None


def startup():
    global model, db
    print("[*] Loading InsightFace model...")
    model = load_model()
    print("[*] Loading database...")
    db = load_database(str(DB_PATH))
    people = list(db["people"].keys())
    print(f"    Known people: {people}")
    for k, v in db["people"].items():
        print(f"      {k}: {v.get('name', '?')} ({v.get('roll_number', '?')})")


startup()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "people_in_db": list(db["people"].keys()) if db else [],
    }


@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...), threshold: float = 0.60):
    if model is None or db is None:
        raise HTTPException(503, "Model or database not loaded")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "Could not decode image")

    emb = get_embedding(model, img)
    if emb is None:
        return {
            "recognized": False,
            "person": None,
            "match_confidence": None,
            "message": "No face detected in the image",
        }

    result = recognize(emb, db, threshold=threshold)
    name = result["name"]

    if name == "UNKNOWN":
        return {
            "recognized": False,
            "person": None,
            "match_confidence": {
                "similarity": result["similarity"],
                "confidence": result["confidence"],
                "cosine_sim": result["cosine_sim"],
                "euclidean_dist": result["euclidean_dist"],
                "chebyshev_dist": result["chebyshev_dist"],
            },
            "message": "Face detected but not matched to any known person",
        }

    person_data = db["people"].get(name, {})
    profile = {
        "name": person_data.get("name", name),
        "roll_number": person_data.get("roll_number", ""),
        "gmail": person_data.get("gmail", ""),
        "program": person_data.get("program", ""),
        "semester": person_data.get("semester", ""),
        "section": person_data.get("section", ""),
        "phone": person_data.get("phone", ""),
        "father_name": person_data.get("father_name", ""),
        "address": person_data.get("address", ""),
    }

    return {
        "recognized": True,
        "person": profile,
        "match_confidence": {
            "similarity": result["similarity"],
            "confidence": result["confidence"],
            "cosine_sim": result["cosine_sim"],
            "euclidean_dist": result["euclidean_dist"],
            "chebyshev_dist": result["chebyshev_dist"],
        },
        "message": "Person identified successfully",
    }


@app.get("/known")
def known_people():
    if db is None:
        raise HTTPException(503, "Database not loaded")
    people = []
    for key, data in db["people"].items():
        people.append({
            "key": key,
            "name": data.get("name", key),
            "roll_number": data.get("roll_number", ""),
            "program": data.get("program", ""),
            "embeddings": data.get("n", 0),
        })
    return {"success": True, "people": people}


@app.post("/register")
async def register_face(
    files: list[UploadFile] = File(...),
    name: str = Form(...),
    roll_number: str = Form(...),
    gmail: str = Form(""),
    program: str = Form(""),
    semester: str = Form(""),
    section: str = Form(""),
    phone: str = Form(""),
    father_name: str = Form(""),
    address: str = Form(""),
):
    if model is None or db is None:
        raise HTTPException(503, "Model or database not loaded")

    person_key = name.lower().replace(" ", "_")

    embeddings = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        emb = get_embedding(model, img)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        raise HTTPException(400, "No valid face detected in any of the uploaded images")

    mean_emb = np.mean(embeddings, axis=0).tolist()
    emb_list = [e.tolist() for e in embeddings]

    student_profile = {
        "name": name,
        "roll_number": roll_number,
        "gmail": gmail,
        "program": program,
        "semester": semester,
        "section": section,
        "phone": phone,
        "father_name": father_name,
        "address": address,
    }

    db["people"][person_key] = {
        "embeddings": emb_list,
        "mean": mean_emb,
        "n": len(embeddings),
        **student_profile,
    }

    # Recompute global stats
    all_embs = []
    for p_data in db["people"].values():
        all_embs.extend(p_data.get("embeddings", []))
    if all_embs:
        all_arr = np.array(all_embs)
        db["stats"]["mean"] = np.mean(all_arr, axis=0).tolist()
        db["stats"]["std"] = np.std(all_arr, axis=0).tolist()

    with open(str(DB_PATH), "wb") as f:
        pickle.dump(db, f)

    print(f"[+] Registered {person_key}: {name} ({roll_number}) — {len(embeddings)} embeddings")

    return {
        "success": True,
        "person_key": person_key,
        "person": student_profile,
        "embeddings_count": len(embeddings),
        "message": f"Successfully registered {name} with {len(embeddings)} face embeddings",
    }


@app.post("/recognize-video")
async def recognize_video(file: UploadFile = File(...), threshold: float = 0.60):
    if model is None or db is None:
        raise HTTPException(503, "Model or database not loaded")

    import tempfile
    import os

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return {"recognized": False, "frames_analyzed": 0, "message": "Could not read video"}

        sample_count = min(5, total)
        indices = [int(i * total / sample_count) for i in range(sample_count)]

        best = {"recognized": False, "person": None, "match_confidence": None, "confidence": 0}

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            emb = get_embedding(model, frame)
            if emb is None:
                continue

            result = recognize(emb, db, threshold=threshold)
            if result["name"] == "UNKNOWN":
                continue

            if result["confidence"] > best["confidence"]:
                person_data = db["people"].get(result["name"], {})
                best = {
                    "recognized": True,
                    "person": {
                        "name": person_data.get("name", result["name"]),
                        "roll_number": person_data.get("roll_number", ""),
                        "gmail": person_data.get("gmail", ""),
                        "program": person_data.get("program", ""),
                        "semester": person_data.get("semester", ""),
                        "section": person_data.get("section", ""),
                        "phone": person_data.get("phone", ""),
                        "father_name": person_data.get("father_name", ""),
                        "address": person_data.get("address", ""),
                    },
                    "match_confidence": {
                        "similarity": result["similarity"],
                        "confidence": result["confidence"],
                        "cosine_sim": result["cosine_sim"],
                        "euclidean_dist": result["euclidean_dist"],
                        "chebyshev_dist": result["chebyshev_dist"],
                    },
                    "confidence": result["confidence"],
                }

        cap.release()
        return {"recognized": best["recognized"], "person": best["person"], "match_confidence": best["match_confidence"], "frames_analyzed": sample_count}
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
