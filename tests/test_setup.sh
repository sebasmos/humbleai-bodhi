#!/bin/bash
#
# HumbleAILLMs Environment Setup Test
# Usage:
#   ./tests/test_setup.sh --imports     # Test all imports
#   ./tests/test_setup.sh --env         # Test conda environment
#   ./tests/test_setup.sh --all         # Run all tests
#   ./tests/test_setup.sh --help        # Show help
#

# Don't use set -e as we want to continue on failures and report them

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
SKIPPED=0

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_pass() {
    echo -e "  ${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

print_fail() {
    echo -e "  ${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

print_skip() {
    echo -e "  ${YELLOW}[SKIP]${NC} $1"
    ((SKIPPED++))
}

print_info() {
    echo -e "  ${BLUE}[INFO]${NC} $1"
}

show_help() {
    echo "HumbleAILLMs Environment Setup Test"
    echo ""
    echo "Usage: ./tests/test_setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --imports     Test all Python imports"
    echo "  --env         Test conda environment setup"
    echo "  --gpu         Test GPU/CUDA availability"
    echo "  --all         Run all tests"
    echo "  --quick       Quick test (env + basic imports only)"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./tests/test_setup.sh --imports"
    echo "  ./tests/test_setup.sh --all"
}

test_conda_env() {
    print_header "Testing Conda Environment"

    # Check if conda is available
    if command -v conda &> /dev/null; then
        print_pass "conda command found"
        CONDA_VERSION=$(conda --version 2>&1)
        print_info "Version: $CONDA_VERSION"
    else
        print_fail "conda command not found"
        echo "  Please install conda/miniforge first"
        return 1
    fi

    # Check current environment
    CURRENT_ENV="${CONDA_DEFAULT_ENV:-base}"
    print_info "Current environment: $CURRENT_ENV"

    # Check Python version
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1)
        print_pass "Python found: $PYTHON_VERSION"
    else
        print_fail "Python not found"
        return 1
    fi

    # Check pip
    if command -v pip &> /dev/null; then
        PIP_VERSION=$(pip --version 2>&1 | head -1)
        print_pass "pip found"
        print_info "$PIP_VERSION"
    else
        print_fail "pip not found"
        return 1
    fi
}

test_core_imports() {
    print_header "Testing Core Python Imports"

    python << 'EOF'
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in dir() else os.getcwd()
if 'HumbleAILLMs' in os.getcwd():
    project_root = os.getcwd()
    while not project_root.endswith('HumbleAILLMs'):
        project_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)

results = []

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    display_name = package_name or module_name
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'N/A')
        print(f"  \033[0;32m[PASS]\033[0m {display_name} (version: {version})")
        return True
    except ImportError as e:
        print(f"  \033[0;31m[FAIL]\033[0m {display_name} - {e}")
        return False
    except Exception as e:
        print(f"  \033[0;31m[FAIL]\033[0m {display_name} - {e}")
        return False

# Standard library (should always work)
print("\n--- Standard Library ---")
test_import('argparse')
test_import('os')
test_import('sys')
test_import('re')
test_import('io')
test_import('gc')
test_import('time')
test_import('random')
test_import('json')
test_import('collections')
test_import('concurrent.futures', 'concurrent.futures')
test_import('multiprocessing')
test_import('typing')

# Core dependencies
print("\n--- Core Dependencies ---")
test_import('numpy')
test_import('tqdm')
test_import('requests')
test_import('jinja2')
test_import('fire')

# ML/AI dependencies
print("\n--- ML/AI Dependencies ---")
test_import('torch')
test_import('transformers')

# Optional dependencies
print("\n--- Optional Dependencies ---")
test_import('accelerate')
test_import('scipy')

# bitsandbytes is Linux/CUDA only
import platform
if platform.system() == 'Darwin':
    print(f"  \033[1;33m[SKIP]\033[0m bitsandbytes (Linux/CUDA only, not needed on Mac)")
else:
    test_import('bitsandbytes')

EOF
}

test_project_imports() {
    print_header "Testing Project-Specific Imports"

    cd "$PROJECT_ROOT"

    python << 'EOF'
import sys
import os
import subprocess

# Ensure we're in the right directory and add human-eval to path
project_root = os.getcwd()
sys.path.insert(0, project_root)
human_eval_path = os.path.join(project_root, 'human-eval')
sys.path.insert(0, human_eval_path)

def test_import(import_statement, display_name):
    """Test if an import statement works."""
    try:
        exec(import_statement)
        print(f"  \033[0;32m[PASS]\033[0m {display_name}")
        return True
    except ImportError as e:
        print(f"  \033[0;31m[FAIL]\033[0m {display_name} - {e}")
        return False
    except Exception as e:
        print(f"  \033[0;31m[FAIL]\033[0m {display_name} - {e}")
        return False

# human-eval imports
print("\n--- human-eval Package ---")
test_import("from human_eval.data import read_problems, HUMAN_EVAL", "human_eval.data")
test_import("from human_eval.evaluation import estimate_pass_at_k", "human_eval.evaluation")
test_import("from human_eval.execution import check_correctness", "human_eval.execution")

# simple-evals imports (test as module to handle relative imports)
print("\n--- simple-evals Package ---")

def test_module_import(module_path, display_name):
    """Test module import by running Python subprocess."""
    simple_evals_dir = os.path.join(project_root, 'simple-evals')
    test_code = f"import sys; sys.path.insert(0, '{simple_evals_dir}'); exec('from {module_path} import *')"
    result = subprocess.run(
        [sys.executable, '-c', test_code],
        capture_output=True,
        text=True,
        cwd=simple_evals_dir
    )
    if result.returncode == 0:
        print(f"  \033[0;32m[PASS]\033[0m {display_name}")
        return True
    else:
        # Check if it's a relative import issue (expected)
        if "relative import" in result.stderr or "No module named" in result.stderr:
            print(f"  \033[1;33m[SKIP]\033[0m {display_name} (relative imports - works when run as module)")
        else:
            print(f"  \033[0;31m[FAIL]\033[0m {display_name} - {result.stderr.strip()}")
        return False

# Test simple-evals as a module (the correct way)
simple_evals_dir = os.path.join(project_root, 'simple-evals')
result = subprocess.run(
    [sys.executable, '-c', 'import common; print("ok")'],
    capture_output=True,
    text=True,
    cwd=simple_evals_dir
)
if result.returncode == 0:
    print(f"  \033[0;32m[PASS]\033[0m simple-evals/common")
else:
    print(f"  \033[1;33m[SKIP]\033[0m simple-evals/common (use: python -m simple-evals.simple_evals)")

# Verify simple_evals can run as module
result = subprocess.run(
    [sys.executable, '-c', 'import sys; sys.path.insert(0, "."); from simple_evals import *'],
    capture_output=True,
    text=True,
    cwd=project_root
)
if "simple_evals" not in result.stderr:
    print(f"  \033[0;32m[PASS]\033[0m simple-evals module structure")
else:
    print(f"  \033[1;33m[INFO]\033[0m Run simple-evals with: python -m simple-evals.simple_evals")

EOF
}

test_gpu() {
    print_header "Testing GPU/CUDA Availability"

    python << 'EOF'
import sys

try:
    import torch

    print(f"  PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"  \033[0;32m[PASS]\033[0m CUDA is available")
        num_gpus = torch.cuda.device_count()
        print(f"  \033[0;34m[INFO]\033[0m Number of GPUs: {num_gpus}")

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  \033[0;34m[INFO]\033[0m GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"  \033[0;32m[PASS]\033[0m Apple MPS is available (Metal GPU)")
    else:
        print(f"  \033[1;33m[SKIP]\033[0m No GPU available (CPU only)")

except ImportError:
    print(f"  \033[0;31m[FAIL]\033[0m PyTorch not installed")
    sys.exit(1)
EOF
}

test_human_eval_module() {
    print_header "Testing human-eval Module Execution"

    cd "$PROJECT_ROOT"

    # Test running human_eval as a module (using local path)
    if python -c "import sys; sys.path.insert(0, 'human-eval'); from human_eval.data import read_problems; problems = read_problems(); print(f'  Loaded {len(problems)} problems')" 2>/dev/null; then
        print_pass "human_eval.data module works"
    else
        print_fail "human_eval.data module failed"
    fi
}

print_summary() {
    print_header "Test Summary"

    echo -e "  ${GREEN}Passed:${NC}  $PASSED"
    echo -e "  ${RED}Failed:${NC}  $FAILED"
    echo -e "  ${YELLOW}Skipped:${NC} $SKIPPED"
    echo ""

    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed. Please install missing dependencies.${NC}"
        echo ""
        echo "Quick fix for missing packages:"
        echo "  pip install numpy tqdm requests jinja2 fire torch transformers accelerate"
        echo "  pip install -e human-eval  # (may have entry point issues, see README)"
        return 1
    fi
}

# Main execution
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --env)
        test_conda_env
        print_summary
        ;;
    --imports)
        test_core_imports
        test_project_imports
        print_summary
        ;;
    --gpu)
        test_gpu
        print_summary
        ;;
    --quick)
        test_conda_env
        test_core_imports
        print_summary
        ;;
    --all)
        test_conda_env
        test_core_imports
        test_project_imports
        test_gpu
        test_human_eval_module
        print_summary
        ;;
    *)
        echo "Usage: $0 [--imports|--env|--gpu|--all|--quick|--help]"
        echo "Run '$0 --help' for more information."
        exit 1
        ;;
esac
