import numpy as np

from ariadne.retrieval.code_hyde import (
	average_hyde_embeddings,
	expand_hyde_queries,
	generate_hyde_code,
)


def test_generate_hyde_code_python():
	code = generate_hyde_code("validate user session token", language="python")
	assert "def validate_user_session_token" in code
	assert "validate user session token" in code
	assert "pass" in code


def test_generate_hyde_code_javascript():
	code = generate_hyde_code("fetch account balance", language="javascript")
	assert "async function fetch_account_balance" in code
	assert "fetch account balance" in code


def test_expand_hyde_queries():
	queries = expand_hyde_queries("hash password bcrypt")
	assert len(queries) == 3
	assert queries[0] == "hash password bcrypt"
	assert "def hash_password_bcrypt" in queries[1]
	assert "function hash_password_bcrypt" in queries[2]


def test_average_hyde_embeddings():
	v1 = np.array([1.0, 0.0, 0.0])
	v2 = np.array([0.0, 1.0, 0.0])
	centroid = average_hyde_embeddings([v1, v2])

	assert np.isclose(np.linalg.norm(centroid), 1.0)
	assert np.isclose(centroid[0], centroid[1])
	assert centroid[2] == 0.0


def test_average_hyde_embeddings_empty():
	res = average_hyde_embeddings([])
	assert len(res) == 0
