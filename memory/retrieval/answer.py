"""turns retrieved facts into an answer, the plain version before chain_of_note.py"""

from datetime import date

ANSWER_PROMPT = """Today's date is {as_of}. Answer the question using only
the facts below. If the facts don't contain the answer, say "I don't know"
— don't guess.

Facts:
{facts_text}

Question: {question}
Answer in one short sentence.
"""

def _format_fact(fact):
    return f"- {fact.subject} {fact.predicate} {fact.object} ({fact.valid_from or '?'} to {fact.valid_to or 'now'})"

def answer_question(question, facts, llm_client, as_of=None, model="claude-sonnet-5"):
    """facts already retrieved by hybrid_search, this just prompts the LLM and returns the text
    as_of is what the question treats as "now", needed for "how long ago" questions"""
    if not facts:
        return "I don't know"

    facts_text = "\n".join(_format_fact(f) for f in facts)
    prompt = ANSWER_PROMPT.format(
        facts_text=facts_text, question=question, as_of=as_of or date.today().isoformat()
    )

    return llm_client.call(model=model, messages=[{"role": "user", "content": prompt}])
