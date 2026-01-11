#!/usr/bin/env python3
"""
EUDEAS Module Tests

Tests the EUDEAS (PRECISE-U + EVS) implementation.

Usage:
    python -m tests.test_eudeas
    python -m tests.test_eudeas --verbose
"""

import argparse
import sys
import os

# Add project paths - simple-evals first to allow relative imports within it
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMPLE_EVALS_DIR = os.path.join(PROJECT_ROOT, 'simple-evals')
sys.path.insert(0, SIMPLE_EVALS_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Colors
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'


def colored(text: str, color: str) -> str:
    return f"{color}{text}{NC}"


def test_components_import():
    """Test that UEUDASComponents can be imported and instantiated."""
    try:
        from eudeas.components import UEUDASComponents

        # Create with default values
        comp = UEUDASComponents()
        assert comp.u_data == 0.0
        assert comp.u_model == 0.0

        # Create with custom values
        comp = UEUDASComponents(
            u_data=0.4,
            u_model=0.3,
            u_ood=0.2,
            u_struct=0.1,
            complexity=0.5,
            confidence=0.6,
            humility=0.7,
            curiosity=0.5,
        )

        # Test total uncertainty
        U = comp.total_uncertainty
        expected_U = 0.3 * 0.4 + 0.3 * 0.3 + 0.2 * 0.2 + 0.2 * 0.1
        assert abs(U - expected_U) < 0.001, f"Expected {expected_U}, got {U}"

        print(f"  {colored('[PASS]', GREEN)} UEUDASComponents instantiation")
        return True
    except Exception as e:
        print(f"  {colored('[FAIL]', RED)} UEUDASComponents instantiation - {e}")
        return False


def test_evs_calculation():
    """Test EVS (Epistemic Virtues Score) calculation."""
    try:
        from eudeas.components import UEUDASComponents

        # Test EVS with matching humility/curiosity to targets
        comp = UEUDASComponents(
            u_data=0.4,
            u_model=0.3,
            u_ood=0.2,
            u_struct=0.2,
            complexity=0.6,
            confidence=0.6,
        )

        # Set humility and curiosity to match targets
        h_star, q_star = comp.targets()
        comp.humility = h_star
        comp.curiosity = q_star

        evs = comp.evs()
        # When H=H* and Q=Q*, EVS should be close to 1.0
        assert 0.9 <= evs <= 1.0, f"Expected EVS near 1.0, got {evs}"

        # Test EVS with mismatched values
        comp.humility = 0.1  # Much lower than target
        evs_low = comp.evs()
        assert evs_low < evs, "EVS should decrease when humility doesn't match target"

        print(f"  {colored('[PASS]', GREEN)} EVS calculation")
        return True
    except Exception as e:
        print(f"  {colored('[FAIL]', RED)} EVS calculation - {e}")
        return False


def test_template_import():
    """Test that PRECISEUTemplate can be imported and used."""
    try:
        from eudeas.template import PRECISEUTemplate

        template = PRECISEUTemplate()

        # Test prompt selection
        uncertainty_profile = {
            'u_data': 0.5,
            'u_model': 0.4,
            'u_ood': 0.3,
            'u_struct': 0.2,
        }
        prompts = template.select_prompts(uncertainty_profile)
        assert len(prompts) > 0, "Should return at least one prompt"
        assert len(prompts) <= 10, "Should not exceed max_prompts"

        # Test template rendering
        case_text = "Patient presents with chest pain and shortness of breath."
        rendered = template.render(case_text, uncertainty_profile)
        assert PRECISEUTemplate.START in rendered
        assert PRECISEUTemplate.END in rendered
        assert "PRECISE-U" in rendered
        assert case_text in rendered

        print(f"  {colored('[PASS]', GREEN)} PRECISEUTemplate")
        return True
    except Exception as e:
        print(f"  {colored('[FAIL]', RED)} PRECISEUTemplate - {e}")
        return False


def test_scorer_import():
    """Test that EUDEASScorer can be imported and used."""
    try:
        from eudeas.scorer import EUDEASScorer, BASIC_WEIGHT, EVS_WEIGHT

        # Verify weights
        assert BASIC_WEIGHT == 0.40
        assert EVS_WEIGHT == 0.60
        assert BASIC_WEIGHT + EVS_WEIGHT == 1.0

        scorer = EUDEASScorer()

        # Test rubric scoring
        completion = """
        P - Probabilistic differential:
        - Acute coronary syndrome: 40%
        - Pulmonary embolism: 25%
        - Other: 35%

        R - Red flags:
        - Chest pain radiating to arm
        - Shortness of breath

        E - Explore uncertainties:
        - Data: 0.4
        - Model: 0.3
        - OOD: 0.2
        - Structural: 0.1

        C - Calculate confidence:
        - Confidence: 60%

        Complexity: 0.6

        Please consult with a physician immediately.
        """
        example = {"prompt": "Patient with chest pain"}

        rubric = scorer.score_rubric(completion, example, is_eudeas_mode=True)
        assert 0.0 <= rubric <= 1.0, f"Rubric score should be in [0,1], got {rubric}"

        print(f"  {colored('[PASS]', GREEN)} EUDEASScorer")
        return True
    except Exception as e:
        print(f"  {colored('[FAIL]', RED)} EUDEASScorer - {e}")
        return False


def test_sampler_wrapper_import():
    """Test that EUDEASSamplerWrapper can be imported."""
    try:
        from eudeas.sampler_wrapper import EUDEASSamplerWrapper

        # Just test import - actual usage requires a real sampler
        assert EUDEASSamplerWrapper is not None

        print(f"  {colored('[PASS]', GREEN)} EUDEASSamplerWrapper import")
        return True
    except Exception as e:
        print(f"  {colored('[FAIL]', RED)} EUDEASSamplerWrapper import - {e}")
        return False


def test_main_module_import():
    """Test that the main eudeas module can be imported."""
    try:
        from eudeas import (
            UEUDASComponents,
            PRECISEUTemplate,
            EUDEASSamplerWrapper,
            EUDEASScorer,
            BASIC_WEIGHT,
            EVS_WEIGHT,
        )

        assert UEUDASComponents is not None
        assert PRECISEUTemplate is not None
        assert EUDEASSamplerWrapper is not None
        assert EUDEASScorer is not None
        assert BASIC_WEIGHT == 0.40
        assert EVS_WEIGHT == 0.60

        print(f"  {colored('[PASS]', GREEN)} Main eudeas module import")
        return True
    except Exception as e:
        print(f"  {colored('[FAIL]', RED)} Main eudeas module import - {e}")
        return False


def test_cli_flag_exists():
    """Test that --use-eudeas flag is recognized by simple_evals."""
    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, '-m', 'simple-evals.simple_evals', '--help'],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        if '--use-eudeas' in result.stdout:
            print(f"  {colored('[PASS]', GREEN)} --use-eudeas CLI flag exists")
            return True
        else:
            print(f"  {colored('[FAIL]', RED)} --use-eudeas CLI flag not found in help")
            return False
    except Exception as e:
        print(f"  {colored('[FAIL]', RED)} CLI flag check - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test EUDEAS module')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    print(colored('=' * 50, BLUE))
    print(colored('EUDEAS Module Tests', BLUE))
    print(colored('=' * 50, BLUE))

    tests = [
        ("Components Import", test_components_import),
        ("EVS Calculation", test_evs_calculation),
        ("Template", test_template_import),
        ("Scorer", test_scorer_import),
        ("Sampler Wrapper Import", test_sampler_wrapper_import),
        ("Main Module Import", test_main_module_import),
        ("CLI Flag", test_cli_flag_exists),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n{colored('Testing:', BLUE)} {name}")
        if test_fn():
            passed += 1
        else:
            failed += 1

    print(f"\n{colored('=' * 50, BLUE)}")
    print(colored('Test Summary', BLUE))
    print(colored('=' * 50, BLUE))
    print(f"  {colored('Passed:', GREEN)} {passed}")
    print(f"  {colored('Failed:', RED)} {failed}")

    if failed == 0:
        print(colored('\nAll EUDEAS tests passed!', GREEN))
        return 0
    else:
        print(colored('\nSome EUDEAS tests failed.', RED))
        return 1


if __name__ == '__main__':
    sys.exit(main())
