"""bi-temporal entity-relation graph: connects facts through shared entities

Edges carry two clocks: valid_from/valid_to (world time) and recorded_at
(when we learned it). recorded_at isn't fed by a real pipeline yet, so
valid_as_of() only answers the world-time question for now.
"""

from datetime import datetime, timezone

import networkx as nx


class EntityGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_fact(self, fact, recorded_at=None):
        """keyed by fact.id + recorded_at, so re-adding the same fact.id after an invalidation adds a parallel edge instead of overwriting"""
        recorded_at = recorded_at or datetime.now(timezone.utc).isoformat()

        self.graph.add_edge(
            fact.subject,
            fact.object,
            key=f"{fact.id}:{recorded_at}",
            fact_id=fact.id,
            predicate=fact.predicate,
            valid_from=fact.valid_from,
            valid_to=fact.valid_to,
            recorded_at=recorded_at,
        )

    def edges_for(self, entity):
        """every edge touching this entity, either direction, with its data"""
        return list(self.graph.in_edges(entity, keys=True, data=True)) + \
               list(self.graph.out_edges(entity, keys=True, data=True))

    def valid_as_of(self, entity, world_time):
        """edges live at world_time: valid_from <= world_time < valid_to (or valid_to is None)"""
        results = []
        for _, _, _, data in self.edges_for(entity):
            vf, vt = data.get("valid_from"), data.get("valid_to")
            if vf and vf > world_time:
                continue
            if vt and vt <= world_time:
                continue
            results.append(data)
        return results
