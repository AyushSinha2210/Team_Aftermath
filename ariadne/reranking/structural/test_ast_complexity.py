from ariadne.reranking.structural.ast_complexity import compute_code_complexity


def test_compute_code_complexity_simple():
	code = """def simple_add(a, b):
    return a + b
"""
	metrics = compute_code_complexity(code)
	assert metrics.cyclomatic_complexity == 1
	assert metrics.branch_count == 0
	assert metrics.line_count == 2
	assert metrics.max_indent_depth == 1


def test_compute_code_complexity_branching():
	code = """def process(items):
    for item in items:
        if item > 0:
            if item % 2 == 0:
                print(item)
        elif item < 0:
            return None
    return True
"""
	metrics = compute_code_complexity(code)
	# branches: for, if, if, elif -> 4 branches
	assert metrics.branch_count >= 4
	assert metrics.cyclomatic_complexity == 1 + metrics.branch_count
	assert metrics.max_indent_depth == 4


def test_compute_code_complexity_empty():
	metrics = compute_code_complexity("")
	assert metrics.cyclomatic_complexity == 1
	assert metrics.line_count == 0
