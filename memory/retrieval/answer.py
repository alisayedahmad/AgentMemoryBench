"""turns retrieved facts into an answer. the plain version of Layer 4's "reading" step, before chain_of_note.py"""

ANSWER_PROMPT = """Answer the question using only the facts below. If the
facts don't contain the answer, say "I don't know" — don't guess.

Facts:
{facts_text}

Question: {question}
Answer in one short sentence.
"""


def _format_fact(fact):
    status = "current" if fact.valid_to is None else f"no longer current as of {fact.valid_to}"
    return f"- {fact.subject} {fact.predicate} {fact.object} ({status})"


def answer_question(question, facts, llm_client, model="claude-sonnet-5"):
    if not facts:
        return "I don't know"

    facts_text = "\n".join(_format_fact(f) for f in facts)
    prompt = ANSWER_PROMPT.format(facts_text=facts_text, question=question)

    return llm_client.call(model=model, messages=[{"role": "user", "content": prompt}])
