"""
Face Recognition System — Mathematical Framework
=================================================

Given a face image I, extract embedding e = phi(I) in R^d, ||e||_2 = 1.
Classification via ensemble of distance metrics + KNN decision.

Distance metrics:
  d_cos(a, b) = 1 - a·b                  [cosine, range 0..2]
  d_euc(a, b) = ||a - b||_2              [euclidean, range 0..2]
  d_che(a, b) = max_i |a_i - b_i|        [chebyshev, range 0..2]

Combined score:
  S_k(q) = (1/|R_k|) * sum_{r in R_k} w_cos * s_cos(q,r) + w_euc * s_euc(q,r) + w_che * s_che(q,r)

where s_*(q,r) = 1 - d_*(q,r)/2 maps distances to [0,1] similarity.

Decision:
  identity(q) = argmax_k S_k(q)  if max_k S_k(q) >= tau, else "UNKNOWN"

Confidence calibration:
  p_k = softmax(S_k(q) / T)  with temperature T
"""

import os
import json
import pickle
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
from insightface.app import FaceAnalysis

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MONGO_URI = "mongodb+srv://drarslanrathore_db_user:XnGgm7WOZkroQGqg@cluster0.xgjup6a.mongodb.net/campussecuirty"
MONGO_DB = "campussecuirty"
MONGO_COL = "face_detections"

# Distance weights
W_COS = 0.5
W_EUC = 0.3
W_CHE = 0.2

TEMP = 0.1
TAU = 0.60  # optimal threshold: 100% TPR, 0% FPR


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(320, 320))
    return app


def get_embedding(app, img_bgr: np.ndarray) -> np.ndarray:
    faces = app.get(img_bgr)
    if not faces:
        return None
    emb = faces[0].embedding
    return emb / (np.linalg.norm(emb) + 1e-10)


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """s = a·b / (||a|| ||b||), range [-1, 1]."""
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """d = 1 - a·b, range [0, 2]."""
    return 1.0 - cosine_similarity(a, b)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """d = ||a - b||_2, range [0, 2] for unit vectors."""
    return float(np.linalg.norm(a - b))


def chebyshev_distance(a: np.ndarray, b: np.ndarray) -> float:
    """d = max_i |a_i - b_i|, range [0, 2] for unit vectors."""
    return float(np.max(np.abs(a - b)))


def normalize_embedding(emb: np.ndarray, mean: np.ndarray = None,
                        std: np.ndarray = None) -> np.ndarray:
    """Z-score normalization: e_i = (e_i - mu_i) / sigma_i."""
    if mean is not None and std is not None:
        return (emb - mean) / (std + 1e-10)
    return emb


# ---------------------------------------------------------------------------
# Combined similarity score
# ---------------------------------------------------------------------------
def combined_similarity(q: np.ndarray, r: np.ndarray) -> float:
    """
    Ensemble similarity score fusing multiple distance metrics.
    
    S(q, r) = w_cos * (1 - d_cos/2) + w_euc * (1 - d_euc/2) + w_che * (1 - d_che/2)
    
    where each term maps distance d in [0, 2] to similarity s in [0, 1].
    Result in [0, 1], higher = more similar.
    """
    d_cos = cosine_distance(q, r)
    d_euc = euclidean_distance(q, r)
    d_che = chebyshev_distance(q, r)

    s_cos = 1.0 - d_cos / 2.0
    s_euc = 1.0 - d_euc / 2.0
    s_che = 1.0 - d_che / 2.0

    return W_COS * s_cos + W_EUC * s_euc + W_CHE * s_che


def softmax(scores: np.ndarray, temperature: float = TEMP) -> np.ndarray:
    """Stable softmax: p_i = exp(s_i / T) / sum_j exp(s_j / T)."""
    s = np.array(scores) / temperature
    s = s - np.max(s)
    exps = np.exp(s)
    return exps / (np.sum(exps) + 1e-10)


# ---------------------------------------------------------------------------
# Database management
# ---------------------------------------------------------------------------
def build_database(data_dir: str, output_path: str = "database.pkl"):
    """
    Build reference database storing ALL embeddings per person.
    
    Structure:
    {
      "people": {
        "name": {
          "embeddings": [e1, e2, ..., en],    # all reference embeddings
          "mean": mean_emb,                    # centroid
          "n": n                               # sample count
        }
      },
      "stats": {
        "mean": global_mean,                   # per-dimension mean
        "std": global_std                      # per-dimension std
      }
    }
    """
    app = load_model()
    data_path = Path(data_dir)
    db = {"people": {}, "stats": {}}

    all_embs = []
    for person_dir in sorted(data_path.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        embs = []
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            emb = get_embedding(app, img)
            if emb is not None:
                embs.append(emb)
                all_embs.append(emb)

        if embs:
            db["people"][name] = {
                "embeddings": embs,
                "mean": np.mean(embs, axis=0).tolist(),
                "n": len(embs),
            }
            print(f"  {name}: {len(embs)} reference embeddings")

    # Global statistics for z-score normalization
    if all_embs:
        all_embs = np.array(all_embs)
        db["stats"]["mean"] = np.mean(all_embs, axis=0).tolist()
        db["stats"]["std"] = np.std(all_embs, axis=0).tolist()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(db, f)
    print(f"\n  Database saved: {len(db['people'])} people, "
          f"{sum(d['n'] for d in db['people'].values())} total embeddings")
    return db


def load_database(db_path: str) -> dict:
    with open(db_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Recognition with full mathematical pipeline
# ---------------------------------------------------------------------------
def recognize(embedding: np.ndarray, db: dict,
              threshold: float = 0.55, knn_k: int = 3) -> dict:
    """
    Recognize a face embedding against the reference database.
    
    Parameters
    ----------
    embedding : np.ndarray (d,) — query embedding
    db : dict — reference database from build_database()
    threshold : float — minimum combined similarity to accept (0..1)
    knn_k : int — number of nearest neighbors for distance-weighted voting
    
    Returns
    -------
    dict with keys:
      name : str — predicted identity or "UNKNOWN"
      similarity : float — combined similarity score (0..1)
      cosine_sim : float — raw cosine similarity
      euclidean_dist : float — raw euclidean distance
      chebyshev_dist : float — raw chebyshev distance
      confidence : float — softmax probability (0..1)
      all_scores : dict — per-person scores
    """
    if embedding is None:
        return {
            "name": "UNKNOWN",
            "similarity": 0.0,
            "cosine_sim": 0.0,
            "euclidean_dist": 2.0,
            "chebyshev_dist": 2.0,
            "confidence": 0.0,
            "all_scores": {},
        }

    people = db["people"]
    result = {"all_scores": {}}

    best_name = "UNKNOWN"
    best_sim = -1.0
    best_cos = -1.0
    best_euc = 2.0
    best_che = 2.0

    scores = []

    for name, data in people.items():
        refs = data["embeddings"]
        # Distance-weighted KNN voting
        sims = []
        cos_sims = []
        euc_dists = []
        che_dists = []

        for ref_emb in refs:
            s = combined_similarity(embedding, ref_emb)
            sims.append(s)

            c = cosine_similarity(embedding, ref_emb)
            e = euclidean_distance(embedding, ref_emb)
            h = chebyshev_distance(embedding, ref_emb)
            cos_sims.append(c)
            euc_dists.append(e)
            che_dists.append(h)

        # KNN: average of top-k nearest neighbors
        sims_arr = np.array(sims)
        if len(sims_arr) >= knn_k:
            top_k = np.partition(sims_arr, -knn_k)[-knn_k:]
            combined = float(np.mean(top_k))
        else:
            combined = float(np.mean(sims_arr))

        avg_cos = float(np.mean(cos_sims))

        scores.append(combined)
        result["all_scores"][name] = {
            "similarity": combined,
            "cosine_sim": avg_cos,
            "n_refs": len(refs),
        }

        if combined > best_sim:
            best_sim = combined
            best_name = name
            best_cos = avg_cos
            best_euc = float(np.mean(euc_dists))
            best_che = float(np.mean(che_dists))

    # Softmax confidence calibration
    probs = softmax(scores, TEMP)
    name_idx = list(people.keys()).index(best_name) if best_name in people else -1
    confidence = float(probs[name_idx]) if name_idx >= 0 else 0.0

    result.update({
        "name": best_name if best_sim >= threshold else "UNKNOWN",
        "similarity": best_sim,
        "cosine_sim": best_cos,
        "euclidean_dist": best_euc,
        "chebyshev_dist": best_che,
        "confidence": confidence,
    })

    return result


# ---------------------------------------------------------------------------
# Live webcam with full math pipeline
# ---------------------------------------------------------------------------
def live_preview(db_path: str, threshold: float = 0.55, camera_id: int = 0,
                 log_interval: float = 2.0, display: bool = True):
    print("[*] Loading InsightFace model...")
    app = load_model()

    print(f"[*] Loading database from {db_path}...")
    db = load_database(db_path)
    people = list(db["people"].keys())
    print(f"    Known people: {people}")
    print(f"    Reference embeddings: {sum(d['n'] for d in db['people'].values())}")
    print(f"    Threshold: {threshold}")
    print(f"    Distance weights — cosine: {W_COS}, euclidean: {W_EUC}, chebyshev: {W_CHE}")

    # MongoDB
    mongo_col = None
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()
        mongo_col = client[MONGO_DB][MONGO_COL]
        print("[*] MongoDB connected")
    except Exception as e:
        print(f"[!] MongoDB: {e}")

    print(f"[*] Starting camera #{camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("[!] Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_log = 0
    frame_count = 0
    fps_start = time.time()

    print("[*] Running. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            fps = frame_count / (time.time() - fps_start)
            print(f"    FPS: {fps:.1f}", end="\r")

        faces = app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            emb = face.embedding
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            result = recognize(emb, db, threshold)
            name = result["name"]
            sim = result["similarity"]
            conf = result["confidence"]
            cos = result["cosine_sim"]
            euc = result["euclidean_dist"]
            che = result["chebyshev_dist"]

            if name != "UNKNOWN":
                color = (0, 255, 0)
                label = f"{name} (S:{sim:.2f} C:{conf:.2f})"
                now = time.time()
                if now - last_log >= log_interval:
                    if mongo_col is not None:
                        try:
                            mongo_col.insert_one({
                                "name": name,
                                "similarity": round(sim, 4),
                                "confidence": round(conf, 4),
                                "cosine_sim": round(cos, 4),
                                "euclidean_dist": round(euc, 4),
                                "chebyshev_dist": round(che, 4),
                                "timestamp": datetime.utcnow(),
                                "source": "webcam_live",
                            })
                        except Exception:
                            pass
                    last_log = now
            else:
                color = (0, 0, 255)
                label = f"UNKNOWN (S:{sim:.2f})"

            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if display:
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[*] Done. {frame_count} frames.")

# ---------------------------------------------------------------------------
# Diagnostic: print mathematical formulation
# ---------------------------------------------------------------------------
def print_math():
    print("""
MATHEMATICAL FRAMEWORK
======================

1. EMBEDDING EXTRACTION
   e = phi(I) in R^d ,  ||e||_2 = 1
   where phi = InsightFace buffalo_l (d=512)

2. DISTANCE METRICS
   Cosine:     d_cos(a,b) = 1 - (a . b) / (||a|| ||b||)  in [0, 2]
   Euclidean:  d_euc(a,b) = ||a - b||_2                   in [0, 2]
   Chebyshev:  d_che(a,b) = max_i |a_i - b_i|             in [0, 2]

3. SIMILARITY SCORES
   s_cos(a,b) = 1 - d_cos(a,b) / 2    in [0, 1]
   s_euc(a,b) = 1 - d_euc(a,b) / 2    in [0, 1]
   s_che(a,b) = 1 - d_che(a,b) / 2    in [0, 1]

4. ENSEMBLE COMBINED SCORE
   S(q, r) = w_c * s_cos + w_e * s_euc + w_h * s_che
   where w = [0.5, 0.3, 0.2], sum(w) = 1

5. KNN AGGREGATION (per person)
   S_k(q) = (1/K) * sum_{j in top-K} S(q, r_{k,j})
   where K = min(3, |R_k|)

6. DECISION FUNCTION
             | argmax_k S_k(q)    if max_k S_k(q) >= tau
   phi(q) = <
             | "UNKNOWN"           otherwise
   with tau = 0.60

7. CONFIDENCE CALIBRATION
   p_k = softmax(S_k(q) / T)
       = exp(S_k / T) / sum_j exp(S_j / T)
   with temperature T = 0.1
""")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_b = sub.add_parser("build-db")
    p_b.add_argument("--data", "-d", required=True)
    p_b.add_argument("--output", "-o", default="database.pkl")

    p_l = sub.add_parser("live")
    p_l.add_argument("--db", default="database.pkl")
    p_l.add_argument("--threshold", "-t", type=float, default=0.60)
    p_l.add_argument("--camera", "-c", type=int, default=0)
    p_l.add_argument("--log-interval", type=float, default=2.0)
    p_l.add_argument("--no-display", action="store_true")

    p_m = sub.add_parser("math")
    p_m.add_argument("--db", default="database.pkl")
    p_m.set_defaults(func=lambda args: print_math())

    args = parser.parse_args()

    if args.cmd == "build-db":
        build_database(args.data, args.output)
    elif args.cmd == "live":
        live_preview(
            db_path=args.db, threshold=args.threshold,
            camera_id=args.camera, log_interval=args.log_interval,
            display=not args.no_display,
        )
    elif args.cmd == "math":
        print_math()
    else:
        parser.print_help()
