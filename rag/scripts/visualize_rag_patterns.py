"""
RAG DB 패턴 시각화
  - rag_viz/class_{name}_sample_{i:02d}.png  (5 classes × 10 samples = 50 files)
  - rag_viz/pca.png  (전체 임베딩 PCA 2D 분포)

실행:
  cd /path/to/cloud-resource-pred
  python scripts/visualize_rag_patterns.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ── 설정 ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("processed")
OUT_DIR = Path("rag_viz")
N_SAMPLES = 10
SEED = 42

CLASS_NAMES = {
    0: "normal-stable",
    1: "normal-rising",
    2: "burst",
    3: "failure-gradual",
    4: "failure-sudden",
}
CLASS_COLORS = {
    0: "#4e79a7",
    1: "#59a14f",
    2: "#f28e2b",
    3: "#e15759",
    4: "#b07aa1",
}

VIZ_FEATURES = [
    (0, "cpu_usage"),
    (6, "cpu_pressure"),
    (8, "cpu_burst"),
]


def extract_stats(x: np.ndarray) -> np.ndarray:
    """x: (N, T, C) → (N, 5*C) [mean, std, min, max, linear_trend]"""
    N, T, C = x.shape
    t_c = np.arange(T, dtype=np.float32) - (T - 1) / 2.0
    t_var = (t_c ** 2).mean()
    mean  = x.mean(axis=1)
    std   = x.std(axis=1)
    xmin  = x.min(axis=1)
    xmax  = x.max(axis=1)
    trend = ((x - mean[:, np.newaxis, :]) * t_c[np.newaxis, :, np.newaxis]).mean(axis=1) / t_var
    return np.concatenate([mean, std, xmin, xmax, trend], axis=1).astype(np.float32)


def denorm(arr: np.ndarray, feat_idx: int, mean: list, std: list) -> np.ndarray:
    return arr * std[feat_idx] + mean[feat_idx]


def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    # Load data
    d = np.load(PROCESSED_DIR / "rag.npz")
    x_raw     = d["x"]           # (N, 32, 10)
    rag_class = d["rag_class"]   # (N,)

    with open(PROCESSED_DIR / "scaler_params.json") as f:
        scaler = json.load(f)
    sc_mean = scaler["mean"]
    sc_std  = scaler["std"]

    timesteps = np.arange(x_raw.shape[1])

    # ── 개별 샘플 PNG ────────────────────────────────────────────────────────
    print("Saving sample PNGs...")
    for cls_id, cls_name in CLASS_NAMES.items():
        indices = np.where(rag_class == cls_id)[0]
        if len(indices) == 0:
            print(f"  [skip] {cls_name}: no samples")
            continue
        chosen = rng.choice(indices, size=min(N_SAMPLES, len(indices)), replace=False)

        for i, idx in enumerate(chosen):
            window = x_raw[idx]  # (32, 10)

            fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
            fig.suptitle(f"Class: {cls_name}  |  Sample {i+1}/{N_SAMPLES}", fontsize=12)

            for ax, (feat_idx, feat_name) in zip(axes, VIZ_FEATURES):
                values = denorm(window[:, feat_idx], feat_idx, sc_mean, sc_std)
                ax.plot(timesteps, values, color=CLASS_COLORS[cls_id], linewidth=1.5)
                ax.set_ylabel(feat_name, fontsize=9)
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("timestep")
            plt.tight_layout()

            fname = OUT_DIR / f"class_{cls_name}_sample_{i:02d}.png"
            fig.savefig(fname, dpi=100)
            plt.close(fig)

        print(f"  {cls_name}: {len(chosen)} files saved")

    # ── PCA 분포 ─────────────────────────────────────────────────────────────
    print("Building PCA plot...")
    embeddings = extract_stats(x_raw)                                   # (N, 50)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings = embeddings / norms

    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(embeddings)                              # (N, 2)
    var_ratio = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls_id, cls_name in CLASS_NAMES.items():
        mask = rag_class == cls_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=3, alpha=0.3, color=CLASS_COLORS[cls_id],
            label=f"{cls_name} (n={mask.sum():,})",
        )

    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_title("RAG DB — Statistical Embedding PCA (47,525 windows)")
    ax.legend(markerscale=4, fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "pca.png", dpi=120)
    plt.close(fig)
    print("  pca.png saved")

    total = sum(1 for _ in OUT_DIR.glob("*.png"))
    print(f"\nDone — {total} PNG files in {OUT_DIR}/")


if __name__ == "__main__":
    main()
