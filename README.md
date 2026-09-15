# AgentMemoryBench

Un système de mémoire à long terme pour les agents IA. Pas un simple cache ou une recherche document, mais une vraie mémoire qui extrait les faits, détecte les contradictions, et les retrouve intelligemment.

## Ce que c'est

Imagine un assistant IA que tu rencontres régulièrement. Aujourd'hui, il oublie tout à chaque conversation. Tu dois expliquer ta carrière, tes préférences, tes changements de vie à chaque fois.

AgentMemoryBench résout ça. C'est une architecture qui:

1. Extrait les faits réels de vos conversations (pas du texte brut)
2. Les nettoie: canonicalize les similaires, détecte les contradictions
3. Les stocke dans un graphe d'entités et une base sémantique
4. Les retrouve avec une recherche hybride (embeddings + keywords + graphe)
5. Les utilise pour répondre sans halluciner

C'est la mémoire humaine: vous vous souvenez des faits, pas de chaque mot. Et vous adaptez votre vision quand quelque chose change.

## Comment ça marche

Exemple simple: un changement de job

```
Episode 1: "Je viens de commencer chez Google"
  -> Extrait: user works_at Google
  -> Stocke le fait

Episode 2: "J'ai quitté Google, maintenant Meta"
  -> Extrait: user works_at Meta
  -> Détecte: contradiction avec le fait précédent
  -> Marque Google comme expiré, ajoute Meta

Question: "Où tu travailles?"
  -> Recherche hybride trouve: works_at Meta
  -> Répond: "Meta"
```

## Architecture

Le code est organisé en 5 étapes:

1. **extraction/** - LLM extrait les faits des conversations
2. **curation/** - Nettoie les faits (canonicalize, dedup, contradictions)
3. **storage/** - Stocke dans SemanticStore (JSONL) et EntityGraph (NetworkX)
4. **retrieval/** - Recherche hybride (dense embeddings + BM25 + graph)
5. **infra/** - Clients LLM, embeddings, cache

## Features principales

- Faits atomiques (subject-predicate-object), pas documents
- Dates ISO normalisées et historique complet (valid_from/valid_to)
- Détection automatique de contradictions
- Recherche hybride: embeddings + BM25 + traversal graphique
- Déduplication sémantique avec threshold
- Oubli naturel (decay exponentiel des vieux faits)
- Consolidation après N apparitions du même fait
- Tests complets, harness d'évaluation

## Installation

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

## Utilisation rapide

```bash
# Test simple avec 4 episodes et 4 questions
python bench/harness.py

# Tests unitaires
pytest -v tests/
```

## État du projet

Complet:
- Extraction, curation, stockage, recherche, oubli naturel
- Suite de tests (environ 20 fichiers)
- Baseline "no memory" pour comparaison

En cours:
- Intégration LongMemEval et LoCoMo
- Métriques complètes (recall, precision, F1)
- Chain-of-Note reasoning

## Fichiers importants

- memory/extraction/extractor.py - Cœur: extraction des faits
- memory/curation/pipeline.py - Orchestrateur du nettoyage
- memory/storage/entity_graph.py - Graphe bi-temporal
- memory/retrieval/hybrid_search.py - Fusion de recherches
- bench/harness.py - Test simple
- infra/llm_client.py - Wrapper Claude

## Pourquoi c'est différent

RAG classique: documents entiers, pas de déduplication sémantique, pas d'historique temporel, pas de détection de contradictions.

AgentMemoryBench: faits atomiques, nettoyage intelligent, bi-temporal, oubli naturel, graphe d'entités, recherche hybride.

## Dépendances

pytest, pydantic, numpy, networkx, rank-bm25, anthropic, sentence-transformers
