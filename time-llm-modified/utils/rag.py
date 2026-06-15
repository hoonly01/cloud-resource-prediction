import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

_FIXED_PAP = (
    "dominant pattern is burst, "
    "failure rate 50% (3 of 5 similar patterns), "
    "Future outlook — "
    "cpu_usage: mean 0.031 max 0.089, "
    "mem_usage: mean 0.041 max 0.112, "
    "cpu_pressure: mean 0.412 max 0.891, "
    "mem_pressure: mean 0.203 max 0.445, "
    "cpu_burst: max 1.834."
)

_CLS_NAMES = {
    0: "normal-stable",
    1: "normal-rising",
    2: "burst",
    3: "failure-gradual",
    4: "failure-sudden",
}


def extract_stats(x: np.ndarray) -> np.ndarray:
    """x: (N, T, C) → (N, 5*C)  [mean, std, min, max, linear_trend]"""
    N, T, C = x.shape
    t_c = np.arange(T, dtype=np.float32) - (T - 1) / 2.0
    t_var = (t_c ** 2).mean()

    mean = x.mean(axis=1)
    std  = x.std(axis=1)
    xmin = x.min(axis=1)
    xmax = x.max(axis=1)
    trend = (
        (x - mean[:, np.newaxis, :]) * t_c[np.newaxis, :, np.newaxis]
    ).mean(axis=1) / t_var

    return np.concatenate([mean, std, xmin, xmax, trend], axis=1).astype(np.float32)


class RAGRetriever:
    def __init__(self, db_dir: str):
        db_dir = Path(db_dir)
        self.index = faiss.read_index(str(db_dir / "faiss.index"))
        meta = np.load(db_dir / "metadata.npz")
        self.y         = meta["y"]          # (N, T, C)
        self.rag_class = meta["rag_class"]  # (N,)
        self.is_fail   = meta["is_fail"]    # (N,)
        with open(db_dir / "db_info.json") as f:
            info = json.load(f)
        self.cpu_u_idx = info.get("cpu_usage_idx", 0)
        self.mem_u_idx = info.get("mem_usage_idx", 2)
        self.cpu_p_idx = info["cpu_pressure_idx"]
        self.mem_p_idx = info.get("mem_pressure_idx", 7)
        self.cpu_b_idx = info["cpu_burst_idx"]

    def query(self, x: np.ndarray, top_k: int = 5, mode: str = 'normal') -> List[str]:
        """
        x: (B, T, C) numpy float32
        mode: 'normal' | 'random' | 'fixed'
          normal — FAISS similarity search (A1)
          random — random sampling from DB, no similarity (A2)
          fixed  — hardcoded PaP for all samples (A3)
        """
        if mode == 'fixed':
            return [_FIXED_PAP] * x.shape[0]

        if mode == 'random':
            idx = np.random.randint(0, len(self.y), size=(x.shape[0], top_k))
            return [self._build_pap(idx[b]) for b in range(x.shape[0])]

        # mode == 'normal'
        emb = extract_stats(x)                                  # (B, 5*C)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        emb = (emb / norms).astype(np.float32)
        _, indices = self.index.search(emb, top_k)             # (B, top_k)
        return [self._build_pap(indices[b]) for b in range(x.shape[0])]

    def _build_pap(self, idx: np.ndarray) -> str:
        y_ret        = self.y[idx]                              # (top_k, T, C)
        fail_rate    = float(self.is_fail[idx].mean())
        n_fail       = int(self.is_fail[idx].sum())
        top_k        = len(idx)
        dominant_cls = _CLS_NAMES[int(np.bincount(self.rag_class[idx]).argmax())]

        cpu_u_mean = float(y_ret[:, :, self.cpu_u_idx].mean())
        cpu_u_max  = float(y_ret[:, :, self.cpu_u_idx].max())
        mem_u_mean = float(y_ret[:, :, self.mem_u_idx].mean())
        mem_u_max  = float(y_ret[:, :, self.mem_u_idx].max())
        cpu_p_mean = float(y_ret[:, :, self.cpu_p_idx].mean())
        cpu_p_max  = float(y_ret[:, :, self.cpu_p_idx].max())
        mem_p_mean = float(y_ret[:, :, self.mem_p_idx].mean())
        mem_p_max  = float(y_ret[:, :, self.mem_p_idx].max())
        cpu_b_max  = float(y_ret[:, :, self.cpu_b_idx].max())

        return (
            f"dominant pattern is {dominant_cls}, "
            f"failure rate {fail_rate:.0%} ({n_fail} of {top_k} similar patterns), "
            f"Future outlook — "
            f"cpu_usage: mean {cpu_u_mean:.3f} max {cpu_u_max:.3f}, "
            f"mem_usage: mean {mem_u_mean:.3f} max {mem_u_max:.3f}, "
            f"cpu_pressure: mean {cpu_p_mean:.3f} max {cpu_p_max:.3f}, "
            f"mem_pressure: mean {mem_p_mean:.3f} max {mem_p_max:.3f}, "
            f"cpu_burst: max {cpu_b_max:.3f}."
        )
