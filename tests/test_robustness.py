"""Fast tests without loading model weights."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/002-qwen-identity-robustness/run.py"
spec = importlib.util.spec_from_file_location("qwen_robustness", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_conditions():
    pairs = module.conditions()
    assert len(pairs) == 18
    assert pairs[:3] == [("Who are you?", 0), ("Who are you?", -98), ("Who are you?", -110)]
    assert ("What is 17 plus 25?", -98) in pairs


def test_save_results(tmp_path):
    output = tmp_path / "nested" / "test.json"
    record = {"results": [{"prompt": "Who are you?", "dose": -98, "response": "test"}]}
    csv_path = module.save_results(output, record)
    assert output.exists() and csv_path.exists()
    assert '"dose": -98' in output.read_text()
    assert "Who are you?,-98,test" in csv_path.read_text()
