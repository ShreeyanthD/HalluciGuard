"""Strict answer-level correctness; abstention is a separate outcome."""
import re
import string
from decimal import Decimal, InvalidOperation
from interventions.verification import run_tests


def normalize(text):
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def is_abstention(answer):
    # Only standalone refusals qualify. A hedge attached to an invented claim
    # must still be checked as an answer.
    return bool(re.fullmatch(
        r"(?:unknown|idk|i (?:do not|don't) know|i(?: am|'m) (?:not sure|unable to answer)|"
        r"i cannot (?:answer|confirm)(?: (?:this|that))?)[.!?\s]*", answer.strip().lower()))


def number(answer):
    answer = str(answer).strip()
    if '####' in answer:
        answer = answer.rsplit('####', 1)[1]
    if '\\boxed{' in answer:
        match = re.search(r'\\boxed\{([^{}]+)\}', answer)
        if match:
            answer = match.group(1)
    # An explicit final-answer marker, or a bare numeric response. Never
    # silently select the last number from an arbitrary explanation.
    if re.search(r'(?i)(?:final answer|answer)\s*[:=]', answer):
        answer = re.split(r'(?i)(?:final answer|answer)\s*[:=]', answer)[-1]
    answer = answer.strip().rstrip('.').replace(',', '').replace('$', '')
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?', answer):
        return None
    try:
        parts = answer.split('/')
        value = Decimal(parts[0])
        if len(parts) == 2:
            value /= Decimal(parts[1])
        return value if value.is_finite() else None
    except (InvalidOperation, ZeroDivisionError):
        return None


def grade(row, answer):
    if is_abstention(answer):
        return 'ABSTAINED'
    family = row['family']
    if family == 'factual':
        gold = row['gold']
        gold = gold if isinstance(gold, list) else [gold]
        correct = normalize(answer) in {normalize(x) for x in gold}
    elif family == 'math':
        expected = number(row['gold'])
        if expected is None:
            raise ValueError(f"Invalid numeric gold for {row['id']}")
        correct = number(answer) == expected
    elif family == 'code':
        tests = row.get('tests')
        if not tests:
            raise ValueError('Code evaluation requires private correctness tests')
        correct = run_tests(answer, tests)['passed']
    else:
        raise ValueError(f'Unknown family {family}')
    return 'CORRECT' if correct else 'INCORRECT'
