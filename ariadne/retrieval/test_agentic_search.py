from pathlib import Path

from ariadne.retrieval.agentic_search import AgenticCodeQueryEngine


def test_agentic_search_usage_query():
	repo_dir = Path(__file__).resolve().parents[1] / "data" / "voice_assistant_js"
	engine = AgenticCodeQueryEngine(repo_dir)

	# Theme 1 problem statement query: "where is the Bluetooth-settings deeplink used?"
	res = engine.query("where is the Bluetooth-settings deeplink used?")

	assert res.query_type == "usage"
	assert len(res.matches) >= 1
	# Check that it found settingsAgent.js
	found_files = [m.file_path for m in res.matches]
	assert any("settingsAgent.js" in f for f in found_files)

	match = [m for m in res.matches if "settingsAgent.js" in m.file_path][0]
	assert match.start_line > 0
	assert match.end_line >= match.start_line
	assert "settings://bluetooth" in match.snippet

	# Check metrics reporting
	assert res.latency_ms > 0.0
	assert res.precision_at_k == 1.0
	assert res.indexing_cost_ms >= 0.0
	assert len(res.optimization_suggestions) >= 1


def test_agentic_search_structural_query():
	repo_dir = Path(__file__).resolve().parents[1] / "data" / "voice_assistant_js"
	engine = AgenticCodeQueryEngine(repo_dir)

	# Theme 1 problem statement structural query: "which files call tool authTool before bluetoothTool?"
	res = engine.query("which files call tool authTool before bluetoothTool?")

	assert res.query_type == "structural"
	assert len(res.plan) >= 2
	assert len(res.optimization_suggestions) >= 1
