"""CPU-only checks for Experiment 012, without loading a language model."""
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/012-qwen-specificity/run.py'
spec = importlib.util.spec_from_file_location('specificity', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.mark.parametrize('task,response,expected', [
    (mod.TASKS[0], '42', True), (mod.TASKS[0], '142', False),
    (mod.TASKS[1], '56.', True), (mod.TASKS[1], 'The answer is 56', True),
    (mod.TASKS[2], 'Paris', True), (mod.TASKS[2], 'Paris is in France', False),
    (mod.TASKS[3], 'Mars.', True),
    (mod.TASKS[4], 'HELLO', True), (mod.TASKS[4], 'hello', False),
    (mod.TASKS[5], 'red, blue, green', True), (mod.TASKS[5], 'blue, red, green', False),
])
def test_scoring(task, response, expected):
    assert mod.score_task(task, response) is expected


def test_input_parsing():
    assert mod.parse_neurons('365,370,386') == [365, 370, 386]
    assert mod.parse_doses('-68,-98') == [-68.0, -98.0]
    with pytest.raises(ValueError):
        mod.parse_neurons('365,365')
    with pytest.raises(ValueError):
        mod.parse_doses('nan')


def test_summary_regression_and_improvement():
    rows = [
        {'task_id': 'a', 'task_type': 'control', 'mode': 'none', 'passed': True},
        {'task_id': 'b', 'task_type': 'control', 'mode': 'none', 'passed': False},
        {'task_id': 'a', 'task_type': 'control', 'mode': 'decode', 'neuron': 386, 'dose': -98., 'passed': False},
        {'task_id': 'b', 'task_type': 'control', 'mode': 'decode', 'neuron': 386, 'dose': -98., 'passed': True},
        {'task_id': 'dev', 'task_type': 'developer', 'mode': 'decode', 'neuron': 386, 'dose': -98., 'attribution': 'Meta'},
    ]
    s = mod.make_summary(rows)[0]
    assert (s['control_regressions'], s['control_improvements'], s['developer_non_alibaba']) == (1, 1, 1)
