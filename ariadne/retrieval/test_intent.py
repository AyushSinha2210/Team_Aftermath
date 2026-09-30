from ariadne.retrieval.intent import (
	QueryIntent,
	classify_query_intent,
)


def test_classify_symbol_lookup():
	res1 = classify_query_intent("get_user_profile")
	assert res1.intent == QueryIntent.SYMBOL_LOOKUP
	assert res1.detected_symbol == "get_user_profile"

	res2 = classify_query_intent("UserAuth.login()")
	assert res2.intent == QueryIntent.SYMBOL_LOOKUP


def test_classify_conceptual():
	res = classify_query_intent("how to implement token refresh in fastapi")
	assert res.intent == QueryIntent.CONCEPTUAL
	assert res.confidence >= 0.8


def test_classify_error_debug():
	res1 = classify_query_intent("TypeError: unsupported operand type for +")
	assert res1.intent == QueryIntent.ERROR_DEBUG
	assert res1.detected_error_pattern == "typeerror"

	res2 = classify_query_intent("assertionerror in unit test pipeline")
	assert res2.intent == QueryIntent.ERROR_DEBUG


def test_classify_empty():
	res = classify_query_intent("  ")
	assert res.intent == QueryIntent.CONCEPTUAL
