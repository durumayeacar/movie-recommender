# src/eval.py (kısa metrik seti)
import numpy as np

def precision_at_k(recommended, ground_truth, k=10):
    rec_k = set(recommended[:k])
    gt = set(ground_truth)
    if not rec_k: return 0.0
    return len(rec_k & gt) / min(k, len(rec_k))

def recall_at_k(recommended, ground_truth, k=10):
    gt = set(ground_truth)
    if not gt: return 0.0
    rec_k = set(recommended[:k])
    return len(rec_k & gt) / len(gt)

def hit_rate_at_k(recommended, ground_truth, k=10):
    return 1.0 if set(recommended[:k]) & set(ground_truth) else 0.0
