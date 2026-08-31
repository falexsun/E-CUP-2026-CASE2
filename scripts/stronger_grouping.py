"""Stronger grouping based on near-text and pHash collisions.

Correct union-find logic:
1. Start with N row nodes (one per product).
2. For each existing entity_group, union all rows in that group FIRST.
3. THEN union collision edge row pairs (near-text, pHash).
4. Final component count MUST be <= original entity_group count.
5. Positive merged components <= positive entity groups.

Near-text edges are label-blind (computed on titles only).
pHash is optional until imagehash is installed.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data


def _norm_title(v):
    v = unicodedata.normalize("NFKC", str(v)).casefold().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", " ", v).strip()


class UnionFind:
    """Weighted quick-union with path compression."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Union two elements. Returns True if they were in different sets."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def n_components(self) -> int:
        return len(set(self.find(i) for i in range(len(self.parent))))


def detect_near_text_collisions(frame: pd.DataFrame, threshold: float = 0.97):
    """Detect near-text collisions using TF-IDF cosine on normalized titles.

    Label-blind: only uses title text, not labels or categories.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    titles = frame["title"].map(_norm_title).values
    categories = frame["category"].values

    collision_pairs = []
    for cat in frame["category"].unique():
        cat_mask = categories == cat
        cat_indices = np.where(cat_mask)[0]
        cat_titles = titles[cat_mask]

        if len(cat_titles) < 2:
            continue

        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=5000)
        tfidf_matrix = tfidf.fit_transform(cat_titles)

        sim = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sim, 0)

        rows, cols = np.where(sim >= threshold)
        for r, c in zip(rows, cols):
            if r < c:
                collision_pairs.append((
                    int(cat_indices[r]),
                    int(cat_indices[c]),
                    float(sim[r, c]),
                ))

    return collision_pairs


def build_merged_groups(
    frame: pd.DataFrame,
    near_text_pairs: list,
    phash_pairs: list | None = None,
) -> tuple[np.ndarray, dict]:
    """Build merged groups using correct union-find logic.

    Step 1: union all rows within each existing entity_group.
    Step 2: union collision edge row pairs.

    Returns (merged_group_labels, stats_dict).
    """
    n = len(frame)
    groups = frame["group"].astype(str).values
    id_to_idx = {str(row["id"]): i for i, row in frame.iterrows()}

    uf = UnionFind(n)

    # Step 1: union rows within each existing entity_group
    group_to_indices = {}
    for i, g in enumerate(groups):
        group_to_indices.setdefault(g, []).append(i)

    orig_group_count = len(group_to_indices)
    intra_unions = 0
    for g, indices in group_to_indices.items():
        for j in range(1, len(indices)):
            if uf.union(indices[0], indices[j]):
                intra_unions += 1

    after_step1 = uf.n_components()
    assert after_step1 == orig_group_count, (
        f"Step 1 failed: expected {orig_group_count} components, got {after_step1}"
    )

    # Step 2: union collision pairs
    near_text_unions = 0
    for idx_a, idx_b, sim in near_text_pairs:
        if uf.union(idx_a, idx_b):
            near_text_unions += 1

    phash_unions = 0
    if phash_pairs:
        for id_a, id_b, h in phash_pairs:
            if id_a in id_to_idx and id_b in id_to_idx:
                if uf.union(id_to_idx[id_a], id_to_idx[id_b]):
                    phash_unions += 1

    final_group_count = uf.n_components()

    # Build labels
    root_to_label = {}
    merged = np.empty(n, dtype=object)
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_label:
            root_to_label[root] = f"mg-{len(root_to_label)}"
        merged[i] = root_to_label[root]

    stats = {
        "n_rows": n,
        "orig_group_count": orig_group_count,
        "final_group_count": final_group_count,
        "intra_unions": intra_unions,
        "near_text_pairs": len(near_text_pairs),
        "near_text_unions": near_text_unions,
        "phash_pairs": len(phash_pairs) if phash_pairs else 0,
        "phash_unions": phash_unions,
    }
    return merged, stats


def main():
    frame, _ = load_data(
        str(ROOT / "data" / "full_grouped.csv"),
        str(ROOT / "configs" / "ozon_schema.json"),
        require_label=True,
    )
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()

    orig_groups = frame["group"].nunique()
    flam_mask = categories == "Легковоспламеняющиеся"
    flam_positive_mask = flam_mask & (labels == 1)
    flam_positive = int(flam_positive_mask.sum())
    flam_orig_groups = frame.loc[flam_positive_mask, "group"].nunique()

    print(f"Rows: {len(frame)}, Original groups: {orig_groups}")
    print(f"Flammable positives: {flam_positive}, Flammable positive groups: {flam_orig_groups}")

    # Detect near-text collisions (label-blind)
    threshold = 0.97
    print(f"\nDetecting near-text collisions (threshold={threshold}, label-blind)...")
    near_text = detect_near_text_collisions(frame, threshold=threshold)
    print(f"  Found {len(near_text)} collision pairs")

    # pHash (optional)
    phash = None
    try:
        import imagehash
        from PIL import Image
        print("\nDetecting pHash collisions...")
        phash = _detect_phash(frame)
        print(f"  Found {len(phash)} pHash collision pairs")
    except ImportError:
        print("\npHash: imagehash not installed, skipping")

    # Build merged groups
    merged, stats = build_merged_groups(frame, near_text, phash)

    # ── invariants ──
    assert stats["final_group_count"] <= stats["orig_group_count"], (
        f"Group count increased: {stats['orig_group_count']} -> {stats['final_group_count']}"
    )
    print(f"\n✓ Group count: {stats['orig_group_count']} -> {stats['final_group_count']} "
          f"(reduced by {stats['orig_group_count'] - stats['final_group_count']})")

    # Positive row count invariant
    assert flam_positive == 198, f"Expected 198 flammable positives, got {flam_positive}"
    print(f"✓ Flammable positives: {flam_positive}")

    # Positive merged components <= positive entity groups
    flam_merged_groups = len(set(merged[flam_positive_mask]))
    assert flam_merged_groups <= flam_positive, (
        f"Positive components exceed positive rows: {flam_merged_groups} > {flam_positive}"
    )
    assert flam_merged_groups <= flam_orig_groups, (
        f"Flammable group count increased: {flam_orig_groups} -> {flam_merged_groups}"
    )
    print(f"✓ Flammable positive groups: {flam_orig_groups} -> {flam_merged_groups} "
          f"(merged {flam_orig_groups - flam_merged_groups})")

    # All invariants pass — save
    out = ROOT / "data" / "full_grouped_merged.csv"
    result = frame.copy()
    result["merged_group"] = merged
    result.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    # Stats
    print(f"\nStats: {stats}")


def _detect_phash(frame):
    """Detect pHash collisions (requires imagehash)."""
    import imagehash
    from PIL import Image

    images_dir = ROOT / "data" / "images"
    hash_map = {}
    for _, row in frame.iterrows():
        product_id = str(row["id"])
        product_dir = images_dir / product_id
        if not product_dir.is_dir():
            continue
        images = sorted(product_dir.glob("*.jpg")) + sorted(product_dir.glob("*.png"))
        if not images:
            continue
        try:
            img = Image.open(images[0]).convert("RGB")
            h = str(imagehash.phash(img))
            img.close()
            hash_map.setdefault(h, []).append(product_id)
        except Exception:
            continue

    pairs = []
    for h, ids in hash_map.items():
        if len(ids) > 1:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pairs.append((ids[i], ids[j], h))
    return pairs


if __name__ == "__main__":
    main()
