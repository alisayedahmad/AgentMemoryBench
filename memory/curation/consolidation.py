"""summarize when fact count per entity grows large, keeping originals in history"""

from datetime import date

from memory.extraction.schemas import Fact

SUMMARY_PROMPT = """Summarize these facts about {subject} into one short sentence:
{lines}
"""

def consolidate(store, subject, llm_client, model="claude-sonnet-5", threshold=5):
    """
    If `subject` has more than `threshold` live facts, asks the LLM for a
    one-sentence summary, adds it as a new fact, and invalidates the
    originals so the current view stays small. Originals are never
    deleted — store.history() still has them. Returns the summary fact,
    or None if the count didn't cross the threshold.
    """
    live = [f for f in store.all() if f.subject == subject and f.valid_to is None]
    if len(live) <= threshold:
        return None

    lines = "\n".join(f"- {f.predicate} {f.object}" for f in live)
    summary_text = llm_client.call(
        model=model,
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(subject=subject, lines=lines)}],
    )

    summary_fact = Fact(
        subject=subject,
        predicate="summary",
        object=summary_text,
        confidence=1.0,
        source_episode_id=live[0].source_episode_id,
        also_seen_in=[f.source_episode_id for f in live[1:]],
        consolidated_from=[f.id for f in live],
    )

    as_of = date.today().isoformat()
    for f in live:
        store.invalidate(f.id, valid_to=as_of)

    store.add(summary_fact)
    return summary_fact
