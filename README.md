# ImageProcessor — Face Recognition System

Real-time face recognition using InsightFace embeddings with ensemble distance metrics, KNN aggregation, and MongoDB logging.

## Mathematical Framework

Given a face image **I**, the model **φ** extracts a unit-normalized embedding on the 512-dimensional hypersphere:

```
e = φ(I) ∈ ℝ⁵¹² ,  ‖e‖₂ = 1
```

### Distance Metrics

| Metric | Equation | Range |
|--------|----------|-------|
| Cosine | `d_cos(a,b) = 1 − a·b` | [0, 2] |
| Euclidean | `d_euc(a,b) = ‖a − b‖₂` | [0, 2] |
| Chebyshev | `d_che(a,b) = maxᵢ |aᵢ − bᵢ|` | [0, 2] |

### Similarity Scores

Each distance is mapped to a similarity in [0, 1]:

```
s_cos(a,b) = 1 − d_cos(a,b) / 2
s_euc(a,b) = 1 − d_euc(a,b) / 2
s_che(a,b) = 1 − d_che(a,b) / 2
```

### Ensemble Score (weighted fusion)

```
S(q, r) = 0.5·s_cos + 0.3·s_euc + 0.2·s_che
```

### Person Score (KNN aggregation, K=3)

```
Sₖ(q) = (1/K) · Σⱼₑₜₒₚ₋ₖ S(q, rₖ,ⱼ)
```

Where Rₖ = {rₖ,₁, ..., rₖ,ₙₖ} is the reference set for person k.

### Decision Function

```
        ⎧ argmaxₖ Sₖ(q)     if maxₖ Sₖ(q) ≥ τ
y(q) =  ⎨
        ⎩ "UNKNOWN"          otherwise
```

τ = 0.60 (optimal — gives 100% TPR, 0% FPR on validation data)

### Confidence Calibration (softmax)

```
pₖ = exp(Sₖ / T) / Σⱼ exp(Sⱼ / T)      T = 0.1
```

## Performance

| Metric | Value |
|--------|-------|
| Intra-class similarity (mean ± std) | 0.8266 ± 0.0797 |
| Inter-class similarity (mean ± std) | 0.5395 ± 0.0202 |
| Separation gap | 0.2871 |
| Unseen image accuracy | 6/6 = 100% |
| Threshold τ = 0.60 | TPR = 1.000, FPR = 0.000 |
| Cross-person max similarity | 0.0764 (faraz vs shahzaib) |

## Project Structure

```
ImageProcessor/
├── face_recognition/
│   ├── __init__.py          # Module exports
│   ├── recognizer.py        # Recognition engine (math, distances, DB)
│   └── collect.py           # Face detection & cropping
├── pipeline.py              # CLI orchestrator
├── assests/                 # Your raw images (organize by person)
│   ├── shahzaib/
│   ├── arslan/
│   └── faraz/
├── data/                    # Generated face crops (gitignored)
├── database.pkl             # Reference embeddings (rebuilt on setup)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The first run will automatically download the InsightFace model (~280MB).

### 2. Prepare face images

Place your images in `assests/<person_name>/`. Example:

```
assests/
├── shahzaib/
│   ├── img1.jpg
│   └── img2.jpg
├── arslan/
│   ├── img1.jpg
│   └── img2.jpg
└── faraz/
    ├── img1.jpg
    └── img2.jpg
```

Need at least 3-5 images per person for robust recognition.

### 3. Run the pipeline

```bash
# Step 1: Detect faces, crop, split into train/val/test
python pipeline.py collect ./assests --output data

# Step 2: Build reference embedding database
python pipeline.py build-db --data data/train

# Step 3: Live webcam recognition
python pipeline.py live --db database.pkl
```

### 4. Additional commands

```bash
# Print the mathematical framework
python pipeline.py math --db database.pkl

# Use a different camera
python pipeline.py live --db database.pkl --camera 1

# Adjust threshold (lower = more sensitive but more false positives)
python pipeline.py live --db database.pkl --threshold 0.50
```

## MongoDB Logging

When a known person is detected, the system logs to MongoDB:

```
database: campussecuirty
collection: face_detections
```

Each document contains:

| Field | Description |
|-------|-------------|
| `name` | Recognized person |
| `similarity` | Ensemble similarity score [0, 1] |
| `confidence` | Softmax probability [0, 1] |
| `cosine_sim` | Raw cosine similarity |
| `euclidean_dist` | Raw euclidean distance |
| `chebyshev_dist` | Raw chebyshev distance |
| `timestamp` | UTC datetime |
| `source` | Always "webcam_live" |

## Adding New People

1. Add images to `assests/<new_name>/`
2. Re-run: `python pipeline.py collect ./assests --output data`
3. Re-build: `python pipeline.py build-db --data data/train`

No retraining needed — embeddings are extracted on the fly.

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit live preview |

## Technical Notes

- **Embedding model**: InsightFace buffalo_l (ResNet100 backbone, trained on 600K faces)
- **Face detection**: InsightFace RetinaFace (ONNX, CPU)
- **Vector dimension**: 512
- **Normalization**: L2 unit norm on all embeddings
- **Threshold selection**: Optimized via Youden's J statistic on validation data
