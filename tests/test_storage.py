from memory.storage.episodic import EpisodicStore
def test_add_returns_id_and_get_retrieves_it(tmp_path):
    store = EpisodicStore(path=str(tmp_path / "episodes.jsonl"))

    episode_id = store.add("I started at Google in March")

    record = store.get(episode_id)
    assert record["text"] == "I started at Google in March"
    assert record["episode_id"] == episode_id


def test_multiple_adds_are_all_kept(tmp_path):
    store = EpisodicStore(path=str(tmp_path / "episodes.jsonl"))

    store.add("first turn")
    store.add("second turn")

    records = store.all()
    assert len(records) == 2
    assert [r["text"] for r in records] == ["first turn", "second turn"]


def test_get_unknown_id_returns_none(tmp_path):
    store = EpisodicStore(path=str(tmp_path / "episodes.jsonl"))

    assert store.get("does-not-exist") is None