"""manual smoke test, not part of pytest. needs a real ANTHROPIC_API_KEY, costs a few cents, safe to rerun (cached)"""

import hashlib

from infra.llm_client import LLMClient
from memory.curation.pipeline import ingest_fact
from memory.extraction.extractor import Extractor
from memory.retrieval.hybrid_search import hybrid_search
from memory.storage.entity_graph import EntityGraph
from memory.storage.semantic import SemanticStore

EPISODES = [
    "I started working at Google last March.",
    "I really love drinking coffee every morning, can't function without it.",
    "Update: I just left Google and joined Meta.",
    "Coffee is honestly the best part of my mornings.",
]

def toy_embed(text, dim=64):
    """fake embedding, word hash only. checks nothing crashes, not real quality"""
    vec = [0.0] * dim
    for word in text.lower().split():
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    return vec

def main():
    llm_client = LLMClient()
    extractor = Extractor(llm_client)
    store = SemanticStore(path="results/raw_outputs/smoke_facts.jsonl")
    graph = EntityGraph()

    for i, text in enumerate(EPISODES):
        episode_id = f"smoke_ep{i}"
        facts = extractor.extract(text, episode_id)
        print(f"\n--- episode {i}: {text!r}")
        if not facts:
            print("  no facts extracted")
            continue
        for fact in facts:
            result, event = ingest_fact(store, graph, fact, embed=toy_embed)
            print(f"  [{event}] {fact.subject} {fact.predicate} {fact.object!r}")

    print("\n--- current facts ---")
    for fact in store.all():
        status = "live" if fact.valid_to is None else f"invalidated ({fact.valid_to})"
        print(f"  {fact.subject} {fact.predicate} {fact.object!r} [{status}]")

    print("\n--- hybrid_search('where does the user work now') ---")
    for fact, score in hybrid_search(store, "where does the user work now", embed=toy_embed, graph=graph):
        print(f"  {score:.3f}  {fact.subject} {fact.predicate} {fact.object!r}")

if __name__ == "__main__":
    main()
