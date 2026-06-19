"""
Face Recognition Module — InsightFace-based recognition system.

Provides face detection, embedding extraction, and identification
using ensemble distance metrics with KNN aggregation.
"""

from .recognizer import (
    load_model, get_embedding,
    build_database, load_database, recognize,
    cosine_similarity, cosine_distance,
    euclidean_distance, chebyshev_distance,
    combined_similarity, softmax,
)
from .collect import organize_dataset
