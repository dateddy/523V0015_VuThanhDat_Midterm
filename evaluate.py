import numpy as np
from scipy.stats import spearmanr

# ── Import everything from the completed skipgram module ──────────────────────
from skipgram import (
    SkipGram, train,
    corpus, pairs, vocab, word2idx, idx2word,
)

# ─────────────────────────────────────────────────────────────────────────────
# Reproduce the baseline-trained model (Task 3.1 spec) so this file is
# self-contained and runnable on its own.
# ─────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE = len(vocab)
EMBED_DIM  = 10
LR_INIT    = 0.025
LR_DECAY   = 0.005
EPOCHS     = 100
SEED       = 0

baseline_model = SkipGram(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                           seed=SEED, init_scale=0.01)
_ = train(baseline_model, pairs, epochs=EPOCHS,
          lr_init=LR_INIT, lr_decay=LR_DECAY)


# ==========================================
# TASK 4.1 (a): Cosine Similarity
# ==========================================
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Returns a scalar in [-1, 1].
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# ==========================================
# TASK 4.1 (b): Word-pair similarities
# ==========================================
WORD_PAIRS = [
    ("cat",     "dog"),
    ("cat",     "mat"),
    ("cat",     "road"),
    ("dog",     "road"),
    ("city",    "road"),
    ("friends", "walk"),
]

def compute_pair_similarities(model: SkipGram, pairs_list: list) -> list:
    """Return list of (w1, w2, similarity) tuples."""
    results = []
    for w1, w2 in pairs_list:
        v1 = model.W_in[word2idx[w1]]
        v2 = model.W_in[word2idx[w2]]
        sim = cosine_similarity(v1, v2)
        results.append((w1, w2, sim))
    # Sort by similarity descending
    results.sort(key=lambda x: x[2], reverse=True)
    return results


# ==========================================
# TASK 4.2 (a): Top-K Nearest Neighbors
# ==========================================
def top_k_neighbors(word: str, k: int, model: SkipGram) -> list:
    """
    Return the k most similar words to `word` (excluding itself),
    using cosine similarity on the learned W_in embeddings.
    Returns list of (neighbor_word, similarity) tuples.
    """
    target_vec = model.W_in[word2idx[word]]
    sims = []
    for other_word, idx in word2idx.items():
        if other_word == word:
            continue
        sim = cosine_similarity(target_vec, model.W_in[idx])
        sims.append((other_word, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]


# ==========================================
# TASK 4.3: Gensim Comparison
# ==========================================
def run_gensim_comparison(pair_sims_scratch: list) -> None:
    """Train Word2Vec with Gensim and compare cosine similarities."""
    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("  [!] gensim not installed — skipping Task 4.3.")
        print("      Install with: pip install gensim")
        return

    sentences = [sentence.lower().split() for sentence in corpus]

    gensim_model = Word2Vec(
        sentences,
        vector_size=10,   # d = 10
        window=2,         # W = 2
        sg=1,             # Skip-gram
        min_count=1,
        workers=1,
        seed=0,
        epochs=100,
    )

    # (a) Gensim cosine similarity for the 6 word pairs
    # Build a lookup: (w1, w2) → scratch sim from the already-sorted list
    scratch_lookup = {(w1, w2): sim for w1, w2, sim in pair_sims_scratch}
    # Unsorted order to match the original table
    gensim_sims = []
    for w1, w2 in WORD_PAIRS:
        try:
            g_sim = float(gensim_model.wv.similarity(w1, w2))
        except KeyError:
            g_sim = float('nan')
        gensim_sims.append((w1, w2, g_sim))

    # Sort by gensim similarity for the extended table
    gensim_sims_sorted = sorted(gensim_sims, key=lambda x: x[2], reverse=True)

    print("\n" + "-" * 75)
    print("TASK 4.3 (a): EXTENDED SIMILARITY TABLE (sorted by Your Value)")
    print("-" * 75)
    print(f"{'Word Pair':<20} {'Your Value':>12} {'Gensim':>10} {'Interpretation'}")
    print("-" * 75)

    # Use the scratch ordering (already sorted by scratch similarity)
    for w1, w2, s_sim in pair_sims_scratch:
        g_sim = scratch_lookup.get((w1, w2),
                next((g for a, b, g in gensim_sims if a == w1 and b == w2), float('nan')))
        # re-fetch gensim sim
        try:
            g_sim = float(gensim_model.wv.similarity(w1, w2))
        except KeyError:
            g_sim = float('nan')
        interp = _interpret(w1, w2, s_sim)
        print(f"({w1}, {w2}){'':<{18 - len(w1) - len(w2) - 4}} {s_sim:>12.4f} {g_sim:>10.4f}   {interp}")

    # (b) Spearman rank correlation
    scratch_order = [sim for _, _, sim in pair_sims_scratch]
    gensim_order  = []
    for w1, w2, _ in pair_sims_scratch:
        try:
            gensim_order.append(float(gensim_model.wv.similarity(w1, w2)))
        except KeyError:
            gensim_order.append(0.0)

    rho, pval = spearmanr(scratch_order, gensim_order)
    print("\n" + "-" * 75)
    print("TASK 4.3 (b): SPEARMAN RANK CORRELATION")
    print("-" * 75)
    print(f"  ρ = {rho:.4f}   (p-value = {pval:.4f})")
    if abs(rho) >= 0.8:
        print("  Interpretation: Strong positive rank agreement — both models")
        print("  assign similar relative orderings to word-pair similarity.")
    elif abs(rho) >= 0.5:
        print("  Interpretation: Moderate rank agreement — orderings partially")
        print("  overlap despite different training objectives.")
    else:
        print("  Interpretation: Weak/no rank agreement — the different training")
        print("  objectives (full softmax vs negative sampling) produce notably")
        print("  different similarity rankings on this small corpus.")

    # (c) Two technical differences
    print("\n" + "-" * 75)
    print("TASK 4.3 (c): TWO TECHNICAL DIFFERENCES")
    print("-" * 75)
    print(
        "  1. Training objective — Our implementation uses FULL SOFTMAX:\n"
        "     the gradient is computed over all V vocabulary words at every\n"
        "     step (O(V) per update). Gensim's Word2Vec uses NEGATIVE\n"
        "     SAMPLING (default negative=5): only the target word and a small\n"
        "     set of randomly-drawn 'noise' words are updated per step\n"
        "     (O(k) ≪ O(V)). This makes Gensim much faster but optimises a\n"
        "     different objective function, leading to different embedding\n"
        "     geometry.\n"
        "\n"
        "  2. Sub-sampling of frequent words — Gensim applies PROBABILISTIC\n"
        "     SUB-SAMPLING to discard occurrences of high-frequency words\n"
        "     (e.g. 'the') with probability proportional to their frequency\n"
        "     (Mikolov et al., 2013). Our implementation trains on every\n"
        "     (center, context) pair without any sub-sampling. On a tiny\n"
        "     corpus like ours this changes which co-occurrence statistics\n"
        "     dominate the embeddings."
    )


def _interpret(w1: str, w2: str, sim: float) -> str:
    """Short semantic interpretation for a word pair."""
    table = {
        ("cat",     "dog"):     "Both are common pets — high similarity expected",
        ("city",    "road"):    "Roads lead to cities — spatial relation",
        ("friends", "walk"):    "'friends walk together' — direct co-occurrence",
        ("cat",     "mat"):     "'cat sat on the mat' — surface co-occurrence",
        ("dog",     "road"):    "'dog ran on the road' — surface co-occurrence",
        ("cat",     "road"):    "Few shared contexts — low similarity expected",
    }
    return table.get((w1, w2), table.get((w2, w1), "—"))


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    # ── Task 4.1 (b) ──────────────────────────────────────────────────────────
    print("=" * 75)
    print("TASK 4.1: COSINE SIMILARITY")
    print("=" * 75)

    pair_sims = compute_pair_similarities(baseline_model, WORD_PAIRS)

    print(f"\n{'Word Pair':<20} {'Your Value':>12}   {'Interpretation'}")
    print("-" * 75)
    for w1, w2, sim in pair_sims:
        interp = _interpret(w1, w2, sim)
        print(f"({w1}, {w2}){'':<{18 - len(w1) - len(w2) - 4}} {sim:>12.4f}   {interp}")

    # ── Task 4.2 (b) ──────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("TASK 4.2: NEAREST NEIGHBORS (Top-3, using W_in embeddings)")
    print("=" * 75)

    query_words = ["cat", "dog", "city", "friends"]

    print(f"\n{'Query':<10} {'Top-3 Neighbors':<35} {'Linguistic Observation'}")
    print("-" * 90)

    observations = {
        "cat":     ("Likely shares context with household/animal words. "
                    "Pairs where neighbours are semantically unexpected reflect "
                    "limited corpus size."),
        "dog":     ("Should be close to 'cat' (both pets appearing together). "
                    "Other neighbours driven by shared sentence contexts."),
        "city":    ("'road' is a strong neighbour ('road leads to the city'). "
                    "Other neighbours may be function words from same sentences."),
        "friends": ("'walk'/'together' expected ('friends walk together every day'). "
                    "Rankings validate direct co-occurrence learning."),
    }

    for qw in query_words:
        neighbors = top_k_neighbors(qw, k=3, model=baseline_model)
        neighbor_str = ", ".join(f"{w}({s:.3f})" for w, s in neighbors)
        obs = observations.get(qw, "—")
        print(f"{qw:<10} {neighbor_str:<35} {obs}")

    # ── Task 4.3 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("TASK 4.3: GENSIM COMPARISON")
    print("=" * 75)
    run_gensim_comparison(pair_sims)