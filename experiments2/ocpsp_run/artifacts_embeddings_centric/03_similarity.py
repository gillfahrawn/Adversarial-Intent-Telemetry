import math

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def euclidean_similarity(a, b):
    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return 1.0 / (1.0 + distance)

def lsh_similarity(a, b):
    matches = sum(1 for x, y in zip(a, b) if (x > 0) == (y > 0))
    return matches / len(a)

def random_projection_hash_similarity(a, b):
    # Simpler bitwise hash similarity based on sign bit
    hash_a = [1 if x >= 0 else 0 for x in a]
    hash_b = [1 if x >= 0 else 0 for x in b]
    matches = sum(1 for x, y in zip(hash_a, hash_b) if x == y)
    return matches / len(a)

def knn_graph_similarity(a, b):
    # Placeholder for graph-based distance; using weighted intersection
    # as an approximation of shared neighbor potential in high-dim space
    intersection = sum(min(abs(x), abs(y)) for x, y in zip(a, b) if (x > 0) == (y > 0))
    union = sum(max(abs(x), abs(y)) for x, y in zip(a, b))
    if union == 0:
        return 0.0
    return intersection / union
