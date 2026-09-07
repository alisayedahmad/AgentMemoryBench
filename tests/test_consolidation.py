from memory.curation.consolidation import consolidate
from memory.extraction.schemas import Fact
from memory.storage.semantic import SemanticStore

class FakeLLMClient:
    def __init__(self, reply="user works in tech and likes coffee"):
        self.reply = reply

    def call(self, model, messages, **kwargs):
        return self.reply

def make_fact(predicate, object, episode="ep1"):
    return Fact(subject="user", predicate=predicate, object=object,
                source_episode_id=episode, confidence=0.9)

def test_below_threshold_does_nothing(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    store.add(make_fact("works_at", "Google"))
    store.add(make_fact("likes", "coffee"))

    result = consolidate(store, "user", FakeLLMClient(), threshold=5)

    assert result is None
    assert len(store.all()) == 2

def test_above_threshold_creates_summary_and_invalidates_originals(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    f1 = make_fact("works_at", "Google", episode="ep1")
    f2 = make_fact("likes", "coffee", episode="ep2")
    f3 = make_fact("lives_in", "Paris", episode="ep3")
    for f in (f1, f2, f3):
        store.add(f)

    result = consolidate(store, "user", FakeLLMClient("summary text"), threshold=2)

    assert result is not None
    assert result.predicate == "summary"
    assert result.object == "summary text"
    assert set(result.consolidated_from) == {f1.id, f2.id, f3.id}

    current = store.find(subject="user")
    live = [f for f in current if f.valid_to is None]
    assert len(live) == 1
    assert live[0].predicate == "summary"

def test_originals_stay_in_history(tmp_path):
    store = SemanticStore(path=str(tmp_path / "facts.jsonl"))
    f1 = make_fact("works_at", "Google")
    f2 = make_fact("likes", "coffee")
    f3 = make_fact("lives_in", "Paris")
    for f in (f1, f2, f3):
        store.add(f)

    consolidate(store, "user", FakeLLMClient(), threshold=2)

    history = store.history(f1.id)
    assert len(history) == 2
    assert history[-1].valid_to is not None
