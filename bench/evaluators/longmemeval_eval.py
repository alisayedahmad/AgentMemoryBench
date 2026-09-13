"""LongMemEval's judge prompts, ported from evaluate_qa.py to call our LLMClient"""

_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information "
    "is not.\n\nQuestion: {question}\n\nExplanation: {answer}\n\n"
    "Model Response: {response}\n\nDoes the model correctly identify the "
    "question as unanswerable? Answer yes or no only."
)

_DEFAULT_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no.{extra}"
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\n"
    "Model Response: {response}\n\nIs the model response correct? Answer yes or no only."
)

_PREFERENCE_TEMPLATE = (
    "I will give you a question, a rubric for desired personalized response, "
    "and a response from a model. Please answer yes if the response satisfies "
    "the desired response. Otherwise, answer no. The model does not need to "
    "reflect all the points in the rubric. The response is correct as long as "
    "it recalls and utilizes the user's personal information correctly."
    "\n\nQuestion: {question}\n\nRubric: {answer}\n\n"
    "Model Response: {response}\n\nIs the model response correct? Answer yes or no only."
)

_TEMPORAL_EXTRA = (
    " In addition, do not penalize off-by-one errors for the number of days. "
    "If the question asks for the number of days/weeks/months, etc., and the "
    "model makes off-by-one errors (e.g., predicting 19 days when the answer "
    "is 18), the model's response is still correct."
)

_KNOWLEDGE_UPDATE_EXTRA = (
    " If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as "
    "the updated answer is the required answer."
)


def _judge_prompt(question_type, question, answer, response, is_abstention):
    if is_abstention:
        return _ABSTENTION_TEMPLATE.format(question=question, answer=answer, response=response)

    if question_type == "single-session-preference":
        return _PREFERENCE_TEMPLATE.format(question=question, answer=answer, response=response)

    extra = ""
    if question_type == "temporal-reasoning":
        extra = _TEMPORAL_EXTRA
    elif question_type == "knowledge-update":
        extra = _KNOWLEDGE_UPDATE_EXTRA

    return _DEFAULT_TEMPLATE.format(question=question, answer=answer, response=response, extra=extra)


def judge_answer(question_type, question, answer, response, llm_client, is_abstention=False,
                 model="claude-sonnet-5", use_cache=True):
    """same judge as their evaluate_qa.py, same prompts, just calling Claude instead of OpenAI
    no temperature knob — the anthropic sdk dropped it, so judge noise gets measured instead of suppressed"""
    prompt = _judge_prompt(question_type, question, answer, response, is_abstention)
    reply = llm_client.call(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=10, use_cache=use_cache,
    )
    return "yes" in reply.lower()
