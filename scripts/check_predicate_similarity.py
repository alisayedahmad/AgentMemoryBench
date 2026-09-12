"""throwaway: prints real cosine similarity between predicate pairs, to set canonicalize_predicate's threshold from data"""

from infra.embedder import make_embedder
from memory.curation.deduplication import cosine_similarity

PAIRS = [
    ("likes", "drinks"),
    ("likes", "hates"),
    ("works_at", "worked_at"),
    ("works_at", "is employed by"),
    ("works_at", "lives_in"),
]

embed = make_embedder()
for a, b in PAIRS:
    print(f"{a!r:20} vs {b!r:20} {cosine_similarity(embed(a), embed(b)):.3f}")
