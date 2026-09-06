"""extracted facts store: persists Facts, supports lookup , and invalidation without deletion"""
import os
from memory.extraction.schemas import Fact
class SemanticStore:
    def __init__(self, path="results/raw_outputs/facts.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def add(self, fact: Fact):
        with open(self.path, "a") as f:
            f.write(fact.model_dump_json() + "\n")

    def _all_versions(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [Fact.model_validate_json(line) for line in f if line.strip()]

    def all(self):
        """current view: latest version of each fact id"""
        latest = {}
        order = []
        for fact in self._all_versions():
            if fact.id not in latest:
                order.append(fact.id)
            latest[fact.id] = fact
        return [latest[fid] for fid in order]

    def history(self, fact_id):
        """every version ever written for one fact id, oldest first"""
        return [f for f in self._all_versions() if f.id == fact_id]

    def find(self, subject=None, predicate=None):
        results = self.all()
        if subject is not None:
            results = [f for f in results if f.subject == subject]
        if predicate is not None:
            results = [f for f in results if f.predicate == predicate]
        return results

    def invalidate(self, fact_id, valid_to):
        """append a new version with valid_to set — never mutates the old line"""
        versions = self.history(fact_id)
        if not versions:
            raise ValueError(f"no fact with id {fact_id}")
        updated = versions[-1].model_copy(update={"valid_to": valid_to})
        self.add(updated)
        return updated