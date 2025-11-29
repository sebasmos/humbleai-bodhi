#!/usr/bin/env python3
"""
Multi-GPU diagnostic tests for HumbleAILLMs.
Run this BEFORE running full evals to catch issues early.

Usage:
    python -m tests.test_multigpu --test all
    python -m tests.test_multigpu --test gpu_detection
    python -m tests.test_multigpu --test small_model
    python -m tests.test_multigpu --test large_model
    python -m tests.test_multigpu --test inference
"""

import argparse
import os
import sys
import time
import gc
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_gpu_detection():
    """Test 1: Basic GPU detection and CUDA availability."""
    print("\n" + "="*60)
    print("TEST 1: GPU Detection")
    print("="*60)

    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available!")
        return False

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs detected: {num_gpus}")

    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")

        # Test basic CUDA operation on each GPU
        try:
            with torch.cuda.device(i):
                x = torch.randn(1000, 1000, device=f"cuda:{i}")
                y = torch.matmul(x, x)
                torch.cuda.synchronize()
            print(f"  Basic CUDA operation: PASS")
        except Exception as e:
            print(f"  Basic CUDA operation: FAIL - {e}")
            return False

    print("\nTEST 1: PASS")
    return True


def test_small_model_loading():
    """Test 2: Load a small model on single GPU."""
    print("\n" + "="*60)
    print("TEST 2: Small Model Loading (GPT-Neo 125M)")
    print("="*60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "EleutherAI/gpt-neo-125m"  # Very small model for quick test

    print(f"Loading {model_id}...")
    start = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda:0"
        )
        load_time = time.time() - start
        print(f"Model loaded in {load_time:.2f}s")

        # Quick inference test
        inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test inference: '{response[:50]}...'")

        # Cleanup
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        print("\nTEST 2: PASS")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multigpu_model_loading(num_gpus: int = 2):
    """Test 3: Load a model across multiple GPUs with device_map='auto'."""
    print("\n" + "="*60)
    print(f"TEST 3: Multi-GPU Model Loading ({num_gpus} GPUs)")
    print("="*60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    available_gpus = torch.cuda.device_count()
    if available_gpus < num_gpus:
        print(f"SKIP: Requested {num_gpus} GPUs but only {available_gpus} available")
        return True  # Not a failure, just skip

    # Use a medium model that benefits from multi-GPU
    # GPT-Neo 1.3B is ~5GB, fits on one GPU but tests the multi-GPU path
    model_id = "EleutherAI/gpt-neo-1.3B"

    print(f"Loading {model_id} with device_map='auto'...")
    print(f"Available GPUs: {available_gpus}, Using: {num_gpus}")

    # Record memory before
    mem_before = []
    for i in range(num_gpus):
        mem_before.append(torch.cuda.memory_allocated(i) / 1024**3)
        print(f"  GPU {i} memory before: {mem_before[i]:.2f} GB")

    start = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        load_time = time.time() - start
        print(f"Model loaded in {load_time:.2f}s")

        # Check memory distribution
        print("\nMemory after loading:")
        for i in range(num_gpus):
            mem_after = torch.cuda.memory_allocated(i) / 1024**3
            print(f"  GPU {i}: {mem_after:.2f} GB (delta: +{mem_after - mem_before[i]:.2f} GB)")

        # Check device_map
        if hasattr(model, 'hf_device_map'):
            unique_devices = set(model.hf_device_map.values())
            print(f"Model spread across devices: {unique_devices}")

        # Inference test
        print("\nRunning inference test...")
        inputs = tokenizer("The capital of France is", return_tensors="pt")
        # Move to first device the model is on
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
        inference_time = time.time() - start

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Inference took {inference_time:.2f}s")
        print(f"Response: '{response}'")

        # Cleanup
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        print("\nTEST 3: PASS")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_large_model_loading(quantize: Optional[str] = None):
    """Test 4: Load Llama-3.3-70B (or similar large model) with multi-GPU."""
    print("\n" + "="*60)
    print(f"TEST 4: Large Model Loading (Llama-3.3-70B)")
    print(f"Quantization: {quantize or 'None (FP16)'}")
    print("="*60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    available_gpus = torch.cuda.device_count()
    if available_gpus < 2:
        print(f"SKIP: Need at least 2 GPUs for 70B model, have {available_gpus}")
        return True

    # Calculate total VRAM
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(available_gpus))
    total_vram_gb = total_vram / 1024**3
    print(f"Total VRAM available: {total_vram_gb:.1f} GB")

    # 70B FP16 needs ~140GB, check if we have enough
    required_vram = 140 if quantize is None else (70 if quantize == "8bit" else 40)
    if total_vram_gb < required_vram:
        print(f"SKIP: Need ~{required_vram}GB VRAM, have {total_vram_gb:.1f}GB")
        return True

    model_id = "meta-llama/Llama-3.3-70B-Instruct"

    # Check HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("WARNING: No HF_TOKEN found. May fail if model requires authentication.")

    print(f"Loading {model_id}...")
    print("This may take 2-5 minutes...")

    # Record memory before
    for i in range(available_gpus):
        mem = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i} memory before: {mem:.2f} GB")

    start = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "token": hf_token,
            "device_map": "auto",
        }

        if quantize == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantize == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        load_time = time.time() - start
        print(f"\nModel loaded in {load_time:.2f}s ({load_time/60:.1f} min)")

        # Check memory distribution
        print("\nMemory after loading:")
        total_used = 0
        for i in range(available_gpus):
            mem = torch.cuda.memory_allocated(i) / 1024**3
            total_used += mem
            print(f"  GPU {i}: {mem:.2f} GB")
        print(f"  Total: {total_used:.2f} GB")

        # Quick inference test
        print("\nRunning inference test...")
        inputs = tokenizer("What is 2+2?", return_tensors="pt")
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        inference_time = time.time() - start

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Inference took {inference_time:.2f}s")
        print(f"Response: '{response[:200]}'")

        # Cleanup
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        print("\nTEST 4: PASS")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_throughput():
    """Test 5: Measure inference throughput to detect hangs."""
    print("\n" + "="*60)
    print("TEST 5: Inference Throughput (Hang Detection)")
    print("="*60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "EleutherAI/gpt-neo-1.3B"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        prompts = [
            "What is the capital of France?",
            "Explain photosynthesis in one sentence.",
            "Write a haiku about programming.",
            "What is 15 * 23?",
            "Name three primary colors.",
        ]

        print(f"\nRunning {len(prompts)} inference requests...")
        times = []

        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt")
            first_device = next(model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}

            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            elapsed = time.time() - start
            times.append(elapsed)

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  [{i+1}/{len(prompts)}] {elapsed:.2f}s - '{response[:50]}...'")

            # Check for potential hang (>30s per request is suspicious)
            if elapsed > 30:
                print(f"  WARNING: Slow inference detected ({elapsed:.1f}s)")

        avg_time = sum(times) / len(times)
        print(f"\nAverage inference time: {avg_time:.2f}s")
        print(f"Total time: {sum(times):.2f}s")

        # Cleanup
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        if avg_time > 30:
            print("\nWARNING: Inference is very slow. Possible issues:")
            print("  - Model not on GPU")
            print("  - Memory swapping")
            print("  - Other processes using GPU")
            return False

        print("\nTEST 5: PASS")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_concurrent_models():
    """Test 6: Test loading two models (simulates eval model + grader model)."""
    print("\n" + "="*60)
    print("TEST 6: Concurrent Models (Eval + Grader)")
    print("="*60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    available_gpus = torch.cuda.device_count()
    if available_gpus < 2:
        print(f"SKIP: Need at least 2 GPUs, have {available_gpus}")
        return True

    # Simulate: small eval model + medium grader model
    eval_model_id = "EleutherAI/gpt-neo-125m"  # Simulates gpt-neo-1.3b
    grader_model_id = "EleutherAI/gpt-neo-1.3B"  # Simulates Llama-70B grader

    try:
        print(f"Loading eval model: {eval_model_id}")
        tokenizer1 = AutoTokenizer.from_pretrained(eval_model_id)
        model1 = AutoModelForCausalLM.from_pretrained(
            eval_model_id,
            torch_dtype=torch.float16,
            device_map="cuda:0"
        )

        print(f"Loading grader model: {grader_model_id}")
        tokenizer2 = AutoTokenizer.from_pretrained(grader_model_id)
        if tokenizer2.pad_token is None:
            tokenizer2.pad_token = tokenizer2.eos_token
        model2 = AutoModelForCausalLM.from_pretrained(
            grader_model_id,
            torch_dtype=torch.float16,
            device_map="auto"  # Will use remaining GPUs
        )

        print("\nMemory usage with both models loaded:")
        for i in range(available_gpus):
            mem = torch.cuda.memory_allocated(i) / 1024**3
            print(f"  GPU {i}: {mem:.2f} GB")

        # Test inference on both
        print("\nTesting inference on eval model...")
        inputs1 = tokenizer1("Hello", return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out1 = model1.generate(**inputs1, max_new_tokens=10)
        print(f"  Eval model response: {tokenizer1.decode(out1[0], skip_special_tokens=True)[:50]}")

        print("Testing inference on grader model...")
        inputs2 = tokenizer2("Hello", return_tensors="pt")
        first_device = next(model2.parameters()).device
        inputs2 = {k: v.to(first_device) for k, v in inputs2.items()}
        with torch.no_grad():
            out2 = model2.generate(**inputs2, max_new_tokens=10, pad_token_id=tokenizer2.eos_token_id)
        print(f"  Grader model response: {tokenizer2.decode(out2[0], skip_special_tokens=True)[:50]}")

        # Cleanup
        del model1, model2, tokenizer1, tokenizer2
        gc.collect()
        torch.cuda.empty_cache()

        print("\nTEST 6: PASS")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU diagnostic tests")
    parser.add_argument(
        "--test",
        type=str,
        default="quick",
        choices=["all", "quick", "gpu_detection", "small_model", "multigpu", "large_model", "inference", "concurrent"],
        help="Which test to run"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs to use for multi-GPU tests"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["4bit", "8bit"],
        help="Quantization for large model test"
    )

    args = parser.parse_args()

    print("="*60)
    print("HumbleAILLMs Multi-GPU Diagnostic Tests")
    print("="*60)
    print(f"Test mode: {args.test}")
    print(f"Num GPUs: {args.num_gpus}")
    print(f"Quantization: {args.quantize or 'None'}")

    results = {}

    if args.test in ["all", "quick", "gpu_detection"]:
        results["gpu_detection"] = test_gpu_detection()

    if args.test in ["all", "quick", "small_model"]:
        results["small_model"] = test_small_model_loading()

    if args.test in ["all", "quick", "multigpu"]:
        results["multigpu"] = test_multigpu_model_loading(args.num_gpus)

    if args.test in ["all", "quick", "inference"]:
        results["inference"] = test_inference_throughput()

    if args.test in ["all", "quick", "concurrent"]:
        results["concurrent"] = test_concurrent_models()

    if args.test in ["all", "large_model"]:
        results["large_model"] = test_large_model_loading(args.quantize)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("="*60)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
