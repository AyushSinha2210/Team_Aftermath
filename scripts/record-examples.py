"""Capture real API responses for the visibly labelled offline safety net."""
import datetime
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen

QUERIES = [
    "How is user input validated before the main handler?",
    "Where is the retry logic for failed requests?",
    "Which functions call the authentication middleware?",
]


def main():
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    records = []
    for query in QUERIES:
        request = Request("http://127.0.0.1:8765/api/search", data=json.dumps({"query": query, "top_k": 3}).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            envelope = json.load(response)
        records.append({"query": query, "response": envelope})
    output = root / "frontend/src/data/recorded-examples.json"
    output.write_text(json.dumps({"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "source_commit": commit, "corpus": "ariadne/data/voice_assistant_js", "records": records}, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(records)} real example responses to {output}")


if __name__ == "__main__":
    main()
