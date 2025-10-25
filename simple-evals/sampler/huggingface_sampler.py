import time
import os
from typing import Any, Optional, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList  # NEW: stopping criteria for early stop
from ..types import MessageList, SamplerBase, SamplerResponse
from dataclasses import dataclass  # NEW: emulate OpenAI usage objects

# NEW: OpenAI-like usage dataclasses
@dataclass  # NEW
class CompletionTokensDetails:  # NEW
    accepted_prediction_tokens: int = 0  # NEW
    audio_tokens: int = 0  # NEW
    reasoning_tokens: int = 0  # NEW
    rejected_prediction_tokens: int = 0  # NEW

@dataclass  # NEW
class PromptTokensDetails:  # NEW
    audio_tokens: int = 0  # NEW
    cached_tokens: int = 0  # NEW

@dataclass  # NEW
class CompletionUsage:  # NEW
    completion_tokens: int  # NEW
    prompt_tokens: int  # NEW
    total_tokens: int  # NEW
    completion_tokens_details: CompletionTokensDetails  # NEW
    prompt_tokens_details: PromptTokensDetails  # NEW

# NEW: stop on dialog-style tokens to avoid chatter that breaks JSON grading
class StopOnSeqs(StoppingCriteria):  # NEW
    def __init__(self, tokenizer, stop_strings: List[str]):  # NEW
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]  # NEW

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # NEW
        seq = input_ids[0].tolist()  # NEW
        for stop in self.stop_ids:  # NEW
            if len(seq) >= len(stop) and seq[-len(stop):] == stop:  # NEW
                return True  # NEW
        return False  # NEW

class HuggingFaceSampler(SamplerBase):
    def __init__(
        self,
        *,
        model_choice: str = "EleutherAI/gpt-neo-1.3B",
        system_message: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        use_chat_template: bool = True,
        truncate_input_tokens: int = 2048,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_choice
        self.system_message = system_message
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.device_pref = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.use_chat_template = use_chat_template
        self.truncate_input_tokens = truncate_input_tokens
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.local_files_only = local_files_only
        self.model = None
        self.tokenizer = None
        self._device_str = "cpu"
        self._load()

    def _handle_text(self, text: str) -> Dict[str, Any]:
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any) -> Dict[str, Any]:
        return {"role": str(role), "content": content}

    def _choose_device(self) -> str:
        if self.device_pref == "cpu":
            return "cpu"
        if self.device_pref == "mps" and torch.backends.mps.is_available():
            return "mps"
        if (self.device_pref in ("auto", "cuda")) and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _infer_dtype(self, device: str) -> torch.dtype:
        if self.torch_dtype is not None:
            return self.torch_dtype
        if device.startswith("cuda"):
            return torch.float16
        if device == "mps":
            return torch.float32
        return torch.float32

    def _load(self) -> None:
        device = self._choose_device()
        torch_dtype = self._infer_dtype(device)
        self._device_str = device

        # Get HuggingFace token from environment (similar to OPENAI_API_KEY)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        quantization_str = ""
        if self.load_in_8bit:
            quantization_str = " (8-bit quantization)"
        elif self.load_in_4bit:
            quantization_str = " (4-bit quantization)"

        print(f"Loading model {self.model_id} on {device} (dtype={torch_dtype}){quantization_str} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            token=hf_token,
            local_files_only=self.local_files_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Check if this is a pre-quantized model (AWQ/GPTQ)
        is_awq = "awq" in self.model_id.lower()
        is_gptq = "gptq" in self.model_id.lower()

        # AWQ/GPTQ models need GPU-only device map (no CPU offloading)
        if is_awq or is_gptq:
            device_map = device if device.startswith("cuda") else "cuda:0"
        else:
            device_map = "auto" if device.startswith("cuda") else None

        # Prepare model loading kwargs
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "token": hf_token,
            "local_files_only": self.local_files_only,
        }

        if self.load_in_8bit:
            # Use BitsAndBytesConfig for 8-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        elif self.load_in_4bit:
            # Use BitsAndBytesConfig for 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,  # Double quantization for better compression
                bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            # For pre-quantized models (AWQ/GPTQ) and normal models
            model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )

        if device_map is None and not (self.load_in_8bit or self.load_in_4bit):
            model = model.to(device)

        self.model = model
        self.tokenizer = tokenizer

        if device.startswith("cuda"):
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"GPU Memory: {gb:.1f} GB")
            if self.load_in_8bit or self.load_in_4bit:
                allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
                print(f"GPU Memory Allocated: {allocated:.1f} GB")
        print("Model loaded successfully.")

    def _build_instruction_prompt(self, messages: MessageList) -> str:
        """Build a simple completion-style prompt that works well with base causal LMs."""
        # Extract just the user messages
        user_texts = []
        for msg in messages:
            if msg.get("role", "user") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_texts.append(content.strip())
                elif isinstance(content, list):
                    # Handle multimodal content
                    text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
                    user_texts.append(" ".join(text_parts).strip())

        user_question = "\n\n".join(user_texts).strip()

        # Simple, direct prompt format that works with base LMs
        prompt = f"""Question: {user_question}

Answer:"""
        return prompt

    def _to_model_input(self, messages: MessageList) -> Dict[str, torch.Tensor]:
        text: str
        if (
            self.use_chat_template
            and hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        ):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self._build_instruction_prompt(messages)  # NEW: use non-dialog instruction prompt
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.truncate_input_tokens,
            padding=False,
        )
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        return inputs

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        trial = 0
        while True:
            try:
                effective_messages: MessageList = list(message_list)
                if self.system_message:
                    effective_messages = [self._pack_message("system", self.system_message)] + effective_messages
                inputs = self._to_model_input(effective_messages)
                temp = float(self.temperature)
                do_sample = temp > 0.0
                top_p = self.top_p
                top_k = 0 if not do_sample else self.top_k
                stop_criteria = StoppingCriteriaList([
                    StopOnSeqs(self.tokenizer, [
                        "\nQuestion:", "\n\nQuestion:",
                        "\nHuman:", "\nUser:", "\nAssistant:",
                        "\n\nHuman:", "\n\nUser:", "\n\nAssistant:",
                        "\nQ:", "\nA:"
                    ])
                ])
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=temp,
                        do_sample=do_sample,
                        top_p=top_p,  # NEW
                        top_k=top_k,  # NEW
                        repetition_penalty=self.repetition_penalty,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        stopping_criteria=stop_criteria,  # NEW
                    )
                input_len = inputs["input_ids"].shape[1]
                response_tokens = outputs[0][input_len:]
                content = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                if not content:
                    content = "[Empty response]"
                prompt_tokens_val = int(input_len)  # NEW: compute prompt_tokens in ints
                completion_tokens_val = int(response_tokens.shape[0])  # NEW: compute completion_tokens in ints
                total_tokens_val = int(prompt_tokens_val + completion_tokens_val)  # NEW: compute total_tokens
                usage_obj = CompletionUsage(  # NEW: build OpenAI-like CompletionUsage
                    completion_tokens=completion_tokens_val,  # NEW
                    prompt_tokens=prompt_tokens_val,  # NEW
                    total_tokens=total_tokens_val,  # NEW
                    completion_tokens_details=CompletionTokensDetails(),  # NEW
                    prompt_tokens_details=PromptTokensDetails(),  # NEW
                )  # NEW
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": usage_obj},  # NEW: only 'usage' to match expected response_dict
                    actual_queried_message_list=effective_messages,
                )
            except torch.cuda.OutOfMemoryError as e:
                print("CUDA OOM encountered; clearing cache and retrying once.")
                torch.cuda.empty_cache()
                if trial == 0:
                    trial += 1
                    continue
                else:
                    usage_obj = CompletionUsage(  # NEW: provide usage object on error
                        completion_tokens=0,  # NEW
                        prompt_tokens=0,  # NEW
                        total_tokens=0,  # NEW
                        completion_tokens_details=CompletionTokensDetails(),  # NEW
                        prompt_tokens_details=PromptTokensDetails(),  # NEW
                    )  # NEW
                    return SamplerResponse(
                        response_text=f"Error: {str(e)}",
                        response_metadata={"usage": usage_obj},  # NEW
                        actual_queried_message_list=effective_messages,
                    )
            except Exception as e:
                backoff = 2 ** trial
                print(f"Exception during generation; retry {trial} after {backoff} sec:", e)
                if trial >= 3:
                    usage_obj = CompletionUsage(  # NEW: provide usage object on error
                        completion_tokens=0,  # NEW
                        prompt_tokens=0,  # NEW
                        total_tokens=0,  # NEW
                        completion_tokens_details=CompletionTokensDetails(),  # NEW
                        prompt_tokens_details=PromptTokensDetails(),  # NEW
                    )  # NEW
                    return SamplerResponse(
                        response_text=f"Error: {str(e)}",
                        response_metadata={"usage": usage_obj},  # NEW
                        actual_queried_message_list=effective_messages,
                    )
                time.sleep(backoff)
                trial += 1
