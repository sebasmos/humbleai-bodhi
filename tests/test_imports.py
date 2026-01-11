#!/usr/bin/env python3
"""
HumbleAILLMs Import Validation Tests

Tests that all required imports for the project work correctly.

Usage:
    python -m tests.test_imports
    python -m tests.test_imports --verbose
    python -m tests.test_imports --category core
"""

import argparse
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'simple-evals'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'human-eval'))


class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class ImportResult:
    name: str
    status: Status
    version: Optional[str] = None
    error: Optional[str] = None


# Color codes
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'


def colored(text: str, color: str) -> str:
    """Return colored text for terminal output."""
    return f"{color}{text}{NC}"


def test_import(module_name: str, from_import: str = None,
                display_name: str = None, optional: bool = False) -> ImportResult:
    """
    Test if a module can be imported.

    Args:
        module_name: The module to import
        from_import: Optional "from X import Y" style import
        display_name: Name to display in output
        optional: If True, mark as SKIP instead of FAIL on error
    """
    name = display_name or module_name

    try:
        if from_import:
            exec(f"from {module_name} import {from_import}")
            return ImportResult(name=name, status=Status.PASS)
        else:
            module = __import__(module_name)
            version = getattr(module, '__version__', None)
            return ImportResult(name=name, status=Status.PASS, version=version)
    except ImportError as e:
        status = Status.SKIP if optional else Status.FAIL
        return ImportResult(name=name, status=status, error=str(e))
    except Exception as e:
        status = Status.SKIP if optional else Status.FAIL
        return ImportResult(name=name, status=status, error=str(e))


def print_result(result: ImportResult, verbose: bool = False):
    """Print a single import result."""
    status_colors = {
        Status.PASS: GREEN,
        Status.FAIL: RED,
        Status.SKIP: YELLOW,
    }

    color = status_colors[result.status]
    status_str = colored(f"[{result.status.value}]", color)

    version_str = f" (v{result.version})" if result.version else ""
    error_str = f" - {result.error}" if result.error and verbose else ""

    print(f"  {status_str} {result.name}{version_str}{error_str}")


def test_standard_library() -> List[ImportResult]:
    """Test standard library imports."""
    modules = [
        'argparse', 'os', 'sys', 're', 'io', 'gc', 'time',
        'random', 'json', 'collections', 'typing', 'dataclasses',
        'multiprocessing', 'pathlib', 'functools', 'itertools'
    ]

    results = []
    for module in modules:
        results.append(test_import(module))

    # Special case for concurrent.futures
    results.append(test_import('concurrent.futures', display_name='concurrent.futures'))

    return results


def test_core_dependencies() -> List[ImportResult]:
    """Test core third-party dependencies."""
    return [
        test_import('numpy'),
        test_import('tqdm'),
        test_import('requests'),
        test_import('jinja2'),
        test_import('fire'),
    ]


def test_ml_dependencies() -> List[ImportResult]:
    """Test ML/AI dependencies."""
    results = [
        test_import('torch'),
        test_import('transformers'),
    ]

    # Test specific transformers imports
    results.append(test_import(
        'transformers',
        from_import='AutoModelForCausalLM, AutoTokenizer',
        display_name='transformers.AutoModelForCausalLM'
    ))

    return results


def test_optional_dependencies() -> List[ImportResult]:
    """Test optional dependencies."""
    return [
        test_import('accelerate', optional=True),
        test_import('bitsandbytes', optional=True),
        test_import('scipy', optional=True),
        test_import('datasets', optional=True),
        test_import('evaluate', optional=True),
    ]


def test_human_eval() -> List[ImportResult]:
    """Test human-eval package imports."""
    results = []

    # Test human_eval imports
    results.append(test_import(
        'human_eval.data',
        from_import='read_problems, HUMAN_EVAL',
        display_name='human_eval.data'
    ))

    results.append(test_import(
        'human_eval.evaluation',
        from_import='estimate_pass_at_k',
        display_name='human_eval.evaluation'
    ))

    results.append(test_import(
        'human_eval.execution',
        from_import='check_correctness',
        display_name='human_eval.execution'
    ))

    return results


def test_simple_evals() -> List[ImportResult]:
    """Test simple-evals package imports."""
    results = []

    # Change to simple-evals directory for relative imports
    original_dir = os.getcwd()
    simple_evals_dir = os.path.join(PROJECT_ROOT, 'simple-evals')

    if os.path.exists(simple_evals_dir):
        os.chdir(simple_evals_dir)
        sys.path.insert(0, '.')

        try:
            results.append(test_import('common', display_name='simple-evals/common'))
            results.append(test_import('types', display_name='simple-evals/types'))
            results.append(test_import('humaneval_eval', display_name='simple-evals/humaneval_eval'))
            results.append(test_import('math_eval', display_name='simple-evals/math_eval'))
            results.append(test_import('mmlu_eval', display_name='simple-evals/mmlu_eval'))
        finally:
            os.chdir(original_dir)
    else:
        results.append(ImportResult(
            name='simple-evals',
            status=Status.SKIP,
            error='Directory not found'
        ))

    return results


def test_gpu_availability() -> List[ImportResult]:
    """Test GPU/CUDA availability."""
    results = []

    try:
        import torch

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_properties(i).name for i in range(num_gpus)]
            results.append(ImportResult(
                name=f'CUDA ({num_gpus} GPUs: {", ".join(gpu_names)})',
                status=Status.PASS
            ))
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            results.append(ImportResult(
                name='Apple MPS (Metal GPU)',
                status=Status.PASS
            ))
        else:
            results.append(ImportResult(
                name='GPU',
                status=Status.SKIP,
                error='No GPU available (CPU only)'
            ))
    except ImportError:
        results.append(ImportResult(
            name='PyTorch',
            status=Status.FAIL,
            error='PyTorch not installed'
        ))

    return results


def run_all_tests(verbose: bool = False, category: str = None) -> Tuple[int, int, int]:
    """
    Run all import tests.

    Returns:
        Tuple of (passed, failed, skipped) counts
    """
    test_categories = {
        'stdlib': ('Standard Library', test_standard_library),
        'core': ('Core Dependencies', test_core_dependencies),
        'ml': ('ML/AI Dependencies', test_ml_dependencies),
        'optional': ('Optional Dependencies', test_optional_dependencies),
        'human_eval': ('human-eval Package', test_human_eval),
        'simple_evals': ('simple-evals Package', test_simple_evals),
        'gpu': ('GPU Availability', test_gpu_availability),
    }

    passed = failed = skipped = 0

    for cat_key, (cat_name, test_fn) in test_categories.items():
        if category and cat_key != category:
            continue

        print(f"\n{colored('─' * 50, BLUE)}")
        print(f"{colored(cat_name, BLUE)}")
        print(colored('─' * 50, BLUE))

        results = test_fn()

        for result in results:
            print_result(result, verbose)
            if result.status == Status.PASS:
                passed += 1
            elif result.status == Status.FAIL:
                failed += 1
            else:
                skipped += 1

    return passed, failed, skipped


def print_summary(passed: int, failed: int, skipped: int):
    """Print test summary."""
    print(f"\n{colored('=' * 50, BLUE)}")
    print(colored('TEST SUMMARY', BLUE))
    print(colored('=' * 50, BLUE))

    print(f"  {colored('Passed:', GREEN)}  {passed}")
    print(f"  {colored('Failed:', RED)}  {failed}")
    print(f"  {colored('Skipped:', YELLOW)} {skipped}")
    print()

    if failed == 0:
        print(colored('All required imports are working!', GREEN))
    else:
        print(colored('Some imports failed. Install missing packages:', RED))
        print()
        print("  pip install numpy tqdm requests jinja2 fire torch transformers")
        print("  pip install accelerate bitsandbytes  # optional, for GPU optimization")


def main():
    parser = argparse.ArgumentParser(
        description='Test HumbleAILLMs import dependencies'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed error messages'
    )
    parser.add_argument(
        '--category', '-c',
        choices=['stdlib', 'core', 'ml', 'optional', 'human_eval', 'simple_evals', 'gpu'],
        help='Test only a specific category'
    )

    args = parser.parse_args()

    print(colored('=' * 50, BLUE))
    print(colored('HumbleAILLMs Import Validation', BLUE))
    print(colored('=' * 50, BLUE))
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project: {PROJECT_ROOT}")

    passed, failed, skipped = run_all_tests(
        verbose=args.verbose,
        category=args.category
    )

    print_summary(passed, failed, skipped)

    sys.exit(1 if failed > 0 else 0)


if __name__ == '__main__':
    main()
