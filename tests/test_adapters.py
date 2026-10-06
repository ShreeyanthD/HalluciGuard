import sys
import types
from data.build import source_rows


def install_dataset(monkeypatch, rows):
    monkeypatch.setitem(sys.modules, 'datasets', types.SimpleNamespace(load_dataset=lambda *a, **k: rows))


def test_sanitized_mbpp_uses_prompt_and_only_contract(monkeypatch):
    install_dataset(monkeypatch, [{'prompt': 'Return sum of two values.',
                    'code': 'def add(a,b):\n return a+b\ndef helper(x):\n return x',
                    'test_list': ['assert add(2,3)==5'], 'test_imports': []}])
    row = source_rows({'dataset': 'mbpp', 'family': 'code', 'adapter': 'mbpp', 'split': 'test'})[0]
    assert 'add(a, b)' in row['question']
    assert 'return a+b' not in row['question'] and '==5' not in row['question']
    assert row['tests'] == ['assert add(2,3)==5']


def test_truthfulqa_aliases_and_math_gold(monkeypatch):
    install_dataset(monkeypatch, [{'question': 'Question', 'correct_answers': ['A', 'B']}])
    row = source_rows({'dataset': 'truthfulqa', 'family': 'factual', 'adapter': 'truthfulqa', 'split': 'validation'})[0]
    assert row['gold'] == ['A', 'B']
    install_dataset(monkeypatch, [{'question': 'Calculate', 'answer': 'work\n#### 42'}])
    row = source_rows({'dataset': 'gsm8k', 'family': 'math', 'adapter': 'gsm8k', 'split': 'train'})[0]
    assert row['gold'].endswith('#### 42')
