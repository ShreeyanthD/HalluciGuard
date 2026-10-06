"""Bounded Python test runner for trusted research inputs.

AST restrictions and subprocess limits are defense in depth, not a secure
sandbox. Use an OS container with networking disabled for untrusted code.
"""
import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED_IMPORTS = {'math', 'typing', 'collections', 'itertools', 'functools', 'statistics',
                   're', 'heapq', 'bisect', 'string', 'fractions', 'decimal', 'operator', 'copy'}
BLOCKED_CALLS = {'exec', 'eval', 'open', 'compile', '__import__', 'getattr', 'setattr',
                 'delattr', 'globals', 'locals', 'vars', 'input', 'breakpoint', 'help'}


def extract_code(answer):
    match = re.search(r'```(?:python)?\s*\n(.*?)```', answer, re.S)
    return match.group(1) if match else answer.strip()


def check_ast(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(n.name not in ALLOWED_IMPORTS for n in node.names):
                raise ValueError('Disallowed import')
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS:
                raise ValueError('Disallowed import')
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('Private attribute access is disallowed')
        if isinstance(node, ast.Name) and (node.id in BLOCKED_CALLS or node.id.startswith('__')):
            raise ValueError('Disallowed name')


def run_tests(answer, tests, timeout=3):
    code = extract_code(answer)
    if isinstance(tests, str):
        tests = [tests]
    # Invalid test definitions are experiment errors, not model failures.
    for test in tests:
        check_ast(test)
    try:
        check_ast(code)
    except (ValueError, SyntaxError) as exc:
        return {'passed': False, 'reason': str(exc)}
    limits = ('import resource\n'
              'resource.setrlimit(resource.RLIMIT_CPU, (2, 2))\n'
              'resource.setrlimit(resource.RLIMIT_AS, (268435456, 268435456))\n'
              'resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))\n')
    with tempfile.TemporaryDirectory(prefix='halluciguard-tests-') as tmp:
        script = Path(tmp) / 'candidate.py'
        script.write_text(limits + code + '\n' + '\n'.join(tests) + '\n')
        try:
            result = subprocess.run([sys.executable, '-I', str(script)], cwd=tmp,
                                    env={'PATH': os.defpath}, timeout=timeout,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {'passed': result.returncode == 0, 'reason': f'exit {result.returncode}'}
        except subprocess.TimeoutExpired:
            return {'passed': False, 'reason': 'timeout'}
