"""append-only timestamped conversation store"""

import json
import os
import uuid
from datetime import datetime, timezone
class EpisodicStore:
    def __init__(self, path="results/raw_outputs/episodes.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def add(self, text, timestamp=None):
        episode_id = str(uuid.uuid4())
        record = {
            "episode_id": episode_id,
            "text": text,
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return episode_id

    def all(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def get(self, episode_id):
        for record in self.all():
            if record["episode_id"] == episode_id:
                return record
        return None