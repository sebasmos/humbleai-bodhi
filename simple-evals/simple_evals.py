import argparse
import json
import subprocess
from datetime import datetime

import pandas as pd

from . import common
from .browsecomp_eval import BrowseCompEval
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .healthbench_eval import HealthBenchEval
from .healthbench_meta_eval import HealthBenchMetaEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .humaneval_eval import HumanEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .sampler.responses_sampler import ResponsesSampler
from .simpleqa_eval import SimpleQAEval

from .sampler.huggingface_sampler import HuggingFaceSampler

def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Select a model by name. Also accepts a comma-separated list of models.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        help="Select an eval by name. Also accepts a comma-separated list of evals.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeats to run. Only supported for certain evals.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=2,
        help="Number of threads to run. Only supported for HealthBench and HealthBenchMeta.",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["4bit", "8bit"],
        help="Quantization to apply to the model (4bit or 8bit).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for model parallelism. Default: use all available GPUs.",
    )

    args = parser.parse_args()

    # Define model_factories as a nested function to capture quantization and num_gpus
    def get_model_factories(quantization: str | None = None, num_gpus: int | None = None):
        """Return dict of model factories with the given quantization and num_gpus settings."""
        return {
            "Llama-3.1-405B-Instruct-FP8": lambda: HuggingFaceSampler(
                model_choice="nvidia/Llama-3.1-405B-Instruct-FP8",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=1024,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "Llama-3.1-405B-FP8": lambda: HuggingFaceSampler(
                model_choice="meta-llama/Llama-3.1-405B-FP8",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=1024,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "mistral-large-instruct": lambda: HuggingFaceSampler(
                model_choice="mistralai/Mistral-Large-Instruct-2407",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=1024,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "mixtral-8x22b-instruct": lambda: HuggingFaceSampler(
                model_choice="mistralai/Mixtral-8x22B-Instruct-v0.1",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            # Local HuggingFace models
            "gpt-neo-1.3b": lambda: HuggingFaceSampler(
                model_choice="EleutherAI/gpt-neo-1.3B",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=1024,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "gpt-oss-120b": lambda: HuggingFaceSampler(
                model_choice="openai/gpt-oss-120b",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=1024,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "gpt-oss-20b": lambda: HuggingFaceSampler(
                model_choice="openai/gpt-oss-20b",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=1024,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "llama-3-70b": lambda: HuggingFaceSampler(
                model_choice="meta-llama/Meta-Llama-3-70B",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "llama-3.3-70b-instruct": lambda: HuggingFaceSampler(
                model_choice="meta-llama/Llama-3.3-70B-Instruct",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "medgemma-4b-it": lambda: HuggingFaceSampler(
                model_choice="google/medgemma-4b-it",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=1024,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "medgemma-4b-pt": lambda: HuggingFaceSampler(
                model_choice="google/medgemma-4b-pt",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=1024,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "medgemma-27b-it": lambda: HuggingFaceSampler(
                model_choice="google/medgemma-27b-it",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=1024,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "medgemma-27b-text-it": lambda: HuggingFaceSampler(
                model_choice="google/medgemma-27b-text-it",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=1024,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            # Qwen and DeepSeek Models
            "qwen3-32b": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen2.5-32B-Instruct",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "deepseek-r1-qwen-32b": lambda: HuggingFaceSampler(
                model_choice="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "qwen2.5-14b-instruct": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen2.5-14B-Instruct",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "qwen3-30b-a3b": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen3-30B-A3B",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "qwen2.5-14b": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen2.5-14B",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "qwen2-72b-instruct": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen2-72B-Instruct",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "qwen2-57b-a14b-instruct": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen2-57B-A14B-Instruct",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            # Qwen Pre-Quantized Models (GPTQ/AWQ - already quantized, loads faster)
            "qwen2.5-3b-instruct-awq": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen2.5-3B-Instruct-AWQ",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "qwen2.5-7b-instruct-awq": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen2.5-7B-Instruct-AWQ",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "qwen2.5-7b-instruct-gptq": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "qwen2.5-14b-instruct-awq": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen2.5-14B-Instruct-AWQ",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "qwen2.5-14b-instruct-gptq-int4": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            "qwen3-30b-a3b-instruct-fp8": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization,
                num_gpus=num_gpus,
            ),
            # Dynamic 4-bit quantization (quantizes on load - needs more RAM initially)
            "qwen2.5-14b-instruct-4bit": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen2.5-14B-Instruct",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization or "4bit",  # Requires ~10-12GB GPU RAM to load, then ~7GB
            ),
            "qwen2.5-14b-4bit": lambda: HuggingFaceSampler(
                model_choice="Qwen/Qwen2.5-14B",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=0.7,
                max_tokens=2048,
                quantize=quantization or "4bit",  # Requires ~10-12GB GPU RAM to load, then ~7GB
            ),
            # Reasoning Models
            "o3": lambda: ResponsesSampler(
                model="o3-2025-04-16",
                reasoning_model=True,
            ),
            "o3-temp-1": lambda: ResponsesSampler(
                model="o3-2025-04-16",
                reasoning_model=True,
                temperature=1.0,
            ),
            "o3_high": lambda: ResponsesSampler(
                model="o3-2025-04-16",
                reasoning_model=True,
                reasoning_effort="high",
            ),
            "o3_low": lambda: ResponsesSampler(
                model="o3-2025-04-16",
                reasoning_model=True,
                reasoning_effort="low",
            ),
            # Default == Medium
            "o4-mini": lambda: ResponsesSampler(
                model="o4-mini-2025-04-16",
                reasoning_model=True,
            ),
            "o4-mini_high": lambda: ResponsesSampler(
                model="o4-mini-2025-04-16",
                reasoning_model=True,
                reasoning_effort="high",
            ),
            "o4-mini_low": lambda: ResponsesSampler(
                model="o4-mini-2025-04-16",
                reasoning_model=True,
                reasoning_effort="low",
            ),
            "o1-pro": lambda: ResponsesSampler(
                model="o1-pro",
                reasoning_model=True,
            ),
            "o1": lambda: OChatCompletionSampler(
                model="o1",
            ),
            "o1_high": lambda: OChatCompletionSampler(
                model="o1",
                reasoning_effort="high",
            ),
            "o1_low": lambda: OChatCompletionSampler(
                model="o1",
                reasoning_effort="low",
            ),
            "o1-preview": lambda: OChatCompletionSampler(
                model="o1-preview",
            ),
            "o1-mini": lambda: OChatCompletionSampler(
                model="o1-mini",
            ),
            # Default == Medium
            "o3-mini": lambda: OChatCompletionSampler(
                model="o3-mini",
            ),
            "o3-mini_high": lambda: OChatCompletionSampler(
                model="o3-mini",
                reasoning_effort="high",
            ),
            "o3-mini_low": lambda: OChatCompletionSampler(
                model="o3-mini",
                reasoning_effort="low",
            ),
            # GPT-4.1 models
            "gpt-4.1": lambda: ChatCompletionSampler(
                model="gpt-4.1-2025-04-14",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
            ),
            "gpt-4.1-temp-1": lambda: ChatCompletionSampler(
                model="gpt-4.1-2025-04-14",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
                temperature=1.0,
            ),
            "gpt-4.1-mini": lambda: ChatCompletionSampler(
                model="gpt-4.1-mini-2025-04-14",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
            ),
            "gpt-4.1-nano": lambda: ChatCompletionSampler(
                model="gpt-4.1-nano-2025-04-14",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
            ),
            # GPT-4o models
            "gpt-4o": lambda: ChatCompletionSampler(
                model="gpt-4o",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
            ),
            "gpt-4o-2024-11-20": lambda: ChatCompletionSampler(
                model="gpt-4o-2024-11-20",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
            ),
            "gpt-4o-2024-08-06": lambda: ChatCompletionSampler(
                model="gpt-4o-2024-08-06",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
            ),
            "gpt-4o-2024-08-06-temp-1": lambda: ChatCompletionSampler(
                model="gpt-4o-2024-08-06",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
                temperature=1.0,
            ),
            "gpt-4o-2024-05-13": lambda: ChatCompletionSampler(
                model="gpt-4o-2024-05-13",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
            ),
            "gpt-4o-mini": lambda: ChatCompletionSampler(
                model="gpt-4o-mini-2024-07-18",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
            ),
            # GPT-4.5 model
            "gpt-4.5-preview": lambda: ChatCompletionSampler(
                model="gpt-4.5-preview-2025-02-27",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
            ),
            # GPT-4-turbo model
            "gpt-4-turbo-2024-04-09": lambda: ChatCompletionSampler(
                model="gpt-4-turbo-2024-04-09",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
            ),
            # GPT-4 model
            "gpt-4-0613": lambda: ChatCompletionSampler(
                model="gpt-4-0613",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
            ),
            # GPT-3.5 Turbo model
            "gpt-3.5-turbo-0125": lambda: ChatCompletionSampler(
                model="gpt-3.5-turbo-0125",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
            ),
            "gpt-3.5-turbo-0125-temp-1": lambda: ChatCompletionSampler(
                model="gpt-3.5-turbo-0125",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=1.0,
            ),
            # Chatgpt models:
            "chatgpt-4o-latest": lambda: ChatCompletionSampler(
                model="chatgpt-4o-latest",
                system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
                max_tokens=2048,
            ),
            "gpt-4-turbo-2024-04-09_chatgpt": lambda: ChatCompletionSampler(
                model="gpt-4-turbo-2024-04-09",
                system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            ),
            # Claude models:
            "claude-3-opus-20240229_empty": lambda: ClaudeCompletionSampler(
                model="claude-3-opus-20240229",
                system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
            ),
            "claude-3-7-sonnet-20250219": lambda: ClaudeCompletionSampler(
                model="claude-3-7-sonnet-20250219",
                system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
            ),
            "claude-3-haiku-20240307": lambda: ClaudeCompletionSampler(
                model="claude-3-haiku-20240307",
            ),
        }

    # Get model factories with the quantization and num_gpus arguments
    model_factories = get_model_factories(args.quantize, args.num_gpus)

    # Auto-generate list of all available models from the factory
    # This ensures the list stays in sync with model_factories
    available_models = list(model_factories.keys())

    if args.list_models:
        print("Available models:")
        for model_name in available_models:
            print(f" - {model_name}")
        return

    if args.model:
        models_chosen = args.model.split(",")
        for model_name in models_chosen:
            if model_name not in available_models:
                print(f"Error: Model '{model_name}' not found.")
                return
        # Lazily initialize only the selected models
        models = {model_name: model_factories[model_name]() for model_name in models_chosen}
    else:
        # If no model specified, initialize all models (original behavior, but lazy)
        models = {model_name: model_factories[model_name]() for model_name in available_models}

    print(f"Running with args {args}")

    # grading_sampler = ChatCompletionSampler(
    #     model="gpt-4.1-2025-04-14",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     max_tokens=2048,
    # )
    # equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")

    # Using local models for grading instead of expensive API calls
    # IMPORTANT: Using AWQ grader to avoid API costs
    # Default: Qwen/Qwen2.5-14B-Instruct-AWQ (~7GB VRAM)
    # For smaller GPUs, swap to Qwen/Qwen2.5-3B-Instruct-AWQ before running
    grading_sampler = HuggingFaceSampler(
        #model_choice="openai/gpt-oss-120b",
        model_choice = "Qwen/Qwen2.5-14B-Instruct-AWQ",
        #model_choice = "Qwen/Qwen2.5-3B-Instruct-AWQ",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        temperature=0.3,  # Lower temperature for more consistent grading
        max_tokens=2048,
    )
    # equality_checker = HuggingFaceSampler(
    #     # model_choice="EleutherAI/gpt-neo-1.3B",
    #     model_choice="openai/gpt-oss-20b",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     temperature=0.1,  # Very low temperature for deterministic yes/no answers
    #     max_tokens=512,   # Shorter max_tokens since it only needs yes/no
    #     device="cpu",  # Force CPU to save GPU memory for evaluated model
    # )
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "mmlu":
                return MMLUEval(num_examples=1 if debug_mode else num_examples)
            # case "math":
            #     return MathEval(
            #         equality_checker=equality_checker,
            #         num_examples=num_examples,
            #         n_repeats=1 if debug_mode else args.n_repeats or 10,
            #     )
            case "gpqa":
                return GPQAEval(
                    n_repeats=1 if debug_mode else args.n_repeats or 10,
                    num_examples=num_examples,
                )
            case "mgsm":
                return MGSMEval(
                    num_examples_per_lang=10 if debug_mode else num_examples or 250
                )
            case "drop":
                return DropEval(
                    num_examples=10 if debug_mode else num_examples,
                    train_samples_per_prompt=3,
                )
            case "humaneval":
                return HumanEval(num_examples=10 if debug_mode else num_examples)
            case "simpleqa":
                return SimpleQAEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case "browsecomp":
                return BrowseCompEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case "healthbench":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name=None,
                )
            case "healthbench_hard":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name="hard",
                )
            case "healthbench_consensus":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name="consensus",
                )
            case "healthbench_meta":
                return HealthBenchMetaEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    if args.eval:
        evals_list = args.eval.split(",")
        evals = {}
        for eval_name in evals_list:
            try:
                evals[eval_name] = get_evals(eval_name, args.debug)
            except Exception as e:
                print(f"Error: eval '{eval_name}' not found.")
                print(f"Exception details: {e}")
                import traceback
                traceback.print_exc()
                return
    else:
        evals = {
            eval_name: get_evals(eval_name, args.debug)
            for eval_name in [
                "mmlu",
                "math",
                "gpqa",
                "mgsm",
                "drop",
                "humaneval",
                "simpleqa",
                "browsecomp",
                "healthbench",
                "healthbench_hard",
                "healthbench_consensus",
                "healthbench_meta",
            ]
        }

    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    print(f"Running the following evals: {list(evals.keys())}")
    print(f"Running evals for the following models: {list(models.keys())}")

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{model_name}"
            # file stem should also include the year, month, day, and time in hours and minutes
            file_stem += f"_{date_str}"
            report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            # Sort metrics by key
            metrics = dict(sorted(metrics.items()))
            print(metrics)
            result_filename = f"/tmp/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")

            full_result_filename = f"/tmp/{file_stem}{debug_suffix}_allresults.json"
            with open(full_result_filename, "w") as f:
                result_dict = {
                    "score": result.score,
                    "metrics": result.metrics,
                    "htmls": result.htmls,
                    "convos": result.convos,
                    "metadata": result.metadata,
                }
                f.write(json.dumps(result_dict, indent=2))
                print(f"Writing all results to {full_result_filename}")

            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
