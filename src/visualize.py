import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from umap import UMAP

# ── Config ───────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR  = "embeddings"
NOTEBOOKS_DIR   = "notebooks"
EMB_FILE        = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
LABELS_FILE     = os.path.join(EMBEDDINGS_DIR, "labels.npy")
PLOT_FILE       = os.path.join(NOTEBOOKS_DIR, "umap_plot.png")

RANDOM_SEED     = 42
UMAP_NEIGHBORS  = 15
UMAP_MIN_DIST   = 0.1

COLORS          = {0: "#3A86FF", 1: "#FF006E"}
LABELS_MAP      = {0: "Transporter", 1: "Kinase"}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

    # Load
    print("\n── Loading embeddings ──")
    X = np.load(EMB_FILE)
    y = np.load(LABELS_FILE)
    print(f"  X shape: {X.shape}  |  Labels: {len(y)}")

    # Run UMAP
    print("\n── Running UMAP (this takes ~1–2 min on CPU) ──")
    reducer = UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        random_state=RANDOM_SEED,
        verbose=True
    )
    X_2d = reducer.fit_transform(X)
    print(f"  UMAP output shape: {X_2d.shape}")

    # Plot
    print("\n── Generating plot ──")
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#0F0F0F")

    for label_val, label_name in LABELS_MAP.items():
        mask = y == label_val
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=COLORS[label_val],
            label=label_name,
            alpha=0.65,
            s=18,
            linewidths=0,
            rasterized=True
        )

    # Title and labels
    ax.set_title(
        "ESM-2 Embedding Space — Kinase vs Transporter",
        fontsize=14,
        color="white",
        pad=15,
        fontweight="bold"
    )
    ax.set_xlabel("UMAP Dimension 1", color="#AAAAAA", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", color="#AAAAAA", fontsize=11)
    ax.tick_params(colors="#555555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # Legend
    legend = ax.legend(
        fontsize=11,
        framealpha=0.2,
        facecolor="#1A1A1A",
        edgecolor="#444444",
        labelcolor="white",
        markerscale=2
    )

    # Annotation
    ax.annotate(
        f"n={len(y)} proteins  |  ESM-2 esm2_t6_8M_UR50D  |  320-dim → UMAP 2D",
        xy=(0.5, -0.08),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        color="#666666"
    )

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"  Saved: {PLOT_FILE}")


if __name__ == "__main__":
    main()