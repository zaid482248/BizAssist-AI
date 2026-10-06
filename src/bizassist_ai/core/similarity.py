"""
Similarity utilities.
Yeh module vectors ke darmiyan similarity calculate karta hai.
"""

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Do vectors ke darmiyan cosine similarity (-1 se 1 tak).
    
    1 = bilkul same direction (most similar)
    0 = perpendicular (unrelated)
   -1 = opposite direction
    
    Args:
        a: First vector.
        b: Second vector.
    
    Returns:
        Cosine similarity as float.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector lengths match nahi: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)