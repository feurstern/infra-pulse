"""The example script and the README's code run against the installed package."""

import re
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_quickstart_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(ROOT / "examples" / "quickstart.py"), run_name="__main__")
    output = capsys.readouterr().out
    assert "missing_value" in output
    assert "timezone_converted" in output
    assert "vps-02" in output


def test_readme_code_prints_what_the_readme_shows(capsys: pytest.CaptureFixture[str]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    code = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    shown = re.findall(r"```text\n(.*?)```", readme, flags=re.DOTALL)
    assert code, "README.md has no python code blocks"
    namespace: dict[str, object] = {"__name__": "__readme__"}
    for block in code:
        exec(compile(block, str(ROOT / "README.md"), "exec"), namespace)
    assert capsys.readouterr().out == shown[0]
