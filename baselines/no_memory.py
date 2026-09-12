"""baseline 1: no memory at all. question goes straight to the LLM, zero context. the floor everything else has to beat"""

NO_MEMORY_PROMPT = """Answer this question in one short sentence. If you
don't know, say "I don't know" — don't guess.

Question: {question}
"""


def answer_with_no_memory(question, llm_client, model="claude-sonnet-5"):
    prompt = NO_MEMORY_PROMPT.format(question=question)
    return llm_client.call(model=model, messages=[{"role": "user", "content": prompt}])
