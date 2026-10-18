"""
End-to-end regression: real questions against a running API.

Skipped unless ASKAI_BASE_URL is set, so `pytest` alone stays fast and offline.

    ASKAI_BASE_URL=http://<vm>:18000 pytest tests/e2e -v

Each case runs in its own session. A case marked "open" in cases.json is a
known failure and reports as xfail (or XPASS once it starts passing — then
flip it to "fixed"). A "fixed" case that fails is a regression.
"""
import json
import os
import re
import uuid
from pathlib import Path

import httpx
import pytest

BASE_URL = os.environ.get("ASKAI_BASE_URL", "").rstrip("/")
TIMEOUT_S = float(os.environ.get("ASKAI_TIMEOUT_S", "180"))
CASES = json.loads((Path(__file__).parent / "cases.json").read_text(encoding="utf-8"))["cases"]

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not BASE_URL, reason="set ASKAI_BASE_URL to run against an API"),
]


def _params():
    for case in CASES:
        marks = [pytest.mark.xfail(reason="known open finding", strict=False)] \
            if case.get("status") == "open" else []
        yield pytest.param(case, id=case["id"], marks=marks)


def _ask(client, session_id, message):
    response = client.post(f"{BASE_URL}/chat", json={"message": message, "session_id": session_id})
    response.raise_for_status()
    return response.json().get("answer") or ""


@pytest.mark.parametrize("case", _params())
def test_case(case):
    session_id = f"e2e-{case['id']}-{uuid.uuid4().hex[:8]}"
    with httpx.Client(timeout=TIMEOUT_S) as client:
        answer = ""
        for turn in case["turns"]:
            answer = _ask(client, session_id, turn)

    missing = [p for p in case.get("must_match", [])
               if not re.search(p, answer, re.IGNORECASE | re.MULTILINE)]
    present = [p for p in case.get("must_not_match", [])
               if re.search(p, answer, re.IGNORECASE | re.MULTILINE)]
    assert not missing and not present, (
        f"\nQ: {case['turns'][-1]}\nA: {answer}\n"
        f"missing: {missing}\nshould not contain: {present}")
