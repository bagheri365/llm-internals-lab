"""CPU-only checks for the localization experiment."""
import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/003-qwen-activation-localization/run.py"
spec = importlib.util.spec_from_file_location("localization", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_conditions():
    assert len(module.conditions()) == 20
    assert module.conditions()[0] == ("Who are you?", "none")


def test_activation_modes_and_no_mutation():
    original = torch.zeros(1, 3, 5)
    for mode in module.MODES:
        result = module.steer_activation(original, 2, -98, mode)
        assert torch.count_nonzero(original) == 0
        expected = {"none": [], "all": [0, 1, 2], "prefill": [0, 1, 2], "last_prompt": [2], "decode": []}[mode]
        assert [i for i in range(3) if result[0, i, 2] != 0] == expected

    decode = torch.zeros(1, 1, 5)
    for mode in module.MODES:
        result = module.steer_activation(decode, 2, -98, mode)
        assert (result[0, 0, 2] != 0).item() == (mode in ("all", "decode"))


def test_save_results(tmp_path):
    output = tmp_path / "nested" / "result.json"
    record = {"results": [{"prompt": "Who are you?", "mode": "all", "dose": -98, "response": "example"}]}
    csv_path = module.save_results(output, record)
    assert output.exists() and csv_path.exists()
    assert "Who are you?,all,-98,example" in csv_path.read_text()
