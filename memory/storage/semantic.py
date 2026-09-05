"""extracted facts store: persists Facts, supports lookup by subject/predicate"""

import os

from memory.extraction.schemas import Fact


class SemanticStore:
    def __init__(self, path="results/raw_outputs/facts.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def add(self, fact: Fact):
        with open(self.path, "a") as f:
            f.write(fact.model_dump_json() + "\n")

    def all(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [Fact.model_validate_json(line) for line in f if line.strip()]

    def find(self, subject=None, predicate=None):
        results = self.all()
        if subject is not None:
            results = [f for f in results if f.subject == subject]
        if predicate is not None:
            results = [f for f in results if f.predicate == predicate]
        return results