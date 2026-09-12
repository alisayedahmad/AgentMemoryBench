"""manual smoke test, not part of pytest. needs a real ANTHROPIC_API_KEY, costs a few cents. first run also downloads a small embedding model (~100MB), then it's cached. wipes its own file each run, don't point this at real data"""

import os

from infra.embedder import make_embedder
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

STORE_PATH = "results/raw_outputs/smoke_facts.jsonl"

def main():
    if os.path.exists(STORE_PATH):
        os.remove(STORE_PATH)

    llm_client = LLMClient()
    extractor = Extractor(llm_client)
    embed = make_embedder()
    store = SemanticStore(path=STORE_PATH)
    graph = EntityGraph()

    for i, text in enumerate(EPISODES):
        episode_id = f"smoke_ep{i}"
        facts = extractor.extract(text, episode_id)
        print(f"\n--- episode {i}: {text!r}")
        if not facts:
            print("  no facts extracted")
            continue
        for fact in facts:
            result, event = ingest_fact(store, graph, fact, embed=embed)
            print(f"  [{event}] {fact.subject} {fact.predicate} {fact.object!r}")

    print("\n--- current facts ---")
    for fact in store.all():
        status = "live" if fact.valid_to is None else f"invalidated ({fact.valid_to})"
        print(f"  {fact.subject} {fact.predicate} {fact.object!r} [{status}]")

    print("\n--- hybrid_search('where does the user work now') ---")
    for fact, score in hybrid_search(store, "where does the user work now", embed=embed, graph=graph):
        print(f"  {score:.3f}  {fact.subject} {fact.predicate} {fact.object!r}")

if __name__ == "__main__":
    main()
