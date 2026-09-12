"""throwaway: same pairs as before, but as full fact sentences instead of bare predicates"""

from infra.embedder import make_embedder
from memory.curation.deduplication import cosine_similarity

PAIRS = [
    ("user likes coffee", "user drinks coffee"),
    ("user likes coffee", "user hates coffee"),
    ("user works_at Google", "user worked_at Google"),
    ("user works_at Google", "user is employed by Google"),
    ("user works_at Google", "user lives_in Google"),
]

embed = make_embedder()
for a, b in PAIRS:
    print(f"{a!r:35} vs {b!r:35} {cosine_similarity(embed(a), embed(b)):.3f}")
