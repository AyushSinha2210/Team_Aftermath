from ariadne.versioning.incremental_sparse import IncrementalSparseIndex


def test_incremental_sparse_insert_and_retrieve():
	index = IncrementalSparseIndex()
	index.insert("d1", "def authenticate_user(username, password): pass")
	index.insert("d2", "def calculate_loss(y_true, y_pred): pass")

	results = index.retrieve("authenticate username", top_k=2)
	assert len(results) >= 1
	assert results[0][0] == "d1"


def test_incremental_sparse_update():
	index = IncrementalSparseIndex()
	index.insert("d1", "initial text apple")
	index.update("d1", "updated text orange")

	assert "d1" in index.docs
	# Should match orange, but not apple
	res_orange = index.retrieve("orange", top_k=1)
	assert len(res_orange) == 1 and res_orange[0][0] == "d1"

	res_apple = index.retrieve("apple", top_k=1)
	assert len(res_apple) == 0


def test_incremental_sparse_remove():
	index = IncrementalSparseIndex()
	index.insert("d1", "def foo(): pass")
	index.insert("d2", "def bar(): pass")

	index.remove("d1")
	assert "d1" not in index.docs
	assert len(index.docs) == 1

	res = index.retrieve("foo", top_k=2)
	assert len(res) == 0
