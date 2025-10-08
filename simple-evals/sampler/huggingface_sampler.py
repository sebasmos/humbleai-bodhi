import time
from typing import Any, Optional, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        use_chat_template: bool = True,
        truncate_input_tokens: int = 800,
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
        print(f"Loading model {self.model_id} on {device} (dtype={torch_dtype}) ...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device_map = "auto" if device.startswith("cuda") else None
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=self.trust_remote_code,
        )
        if device_map is None:
            model = model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        if device.startswith("cuda"):
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"GPU Memory: {gb:.1f} GB")
        print("Model loaded successfully.")

    def _build_instruction_prompt(self, messages: MessageList) -> str:  # NEW: instruction-style fallback prompt
        system_text = self.system_message or "You are a helpful assistant."  # NEW
        parts: List[str] = []  # NEW
        parts.append(  # NEW
            "System:\n"
            f"{system_text}\n\n"
            "Instructions:\n"
            "- Write a single, self-contained answer for the user.\n"
            "- Use concise, professional Markdown.\n"
            "- Do NOT include any dialog tags like 'Human:' or 'Assistant:'.\n"
            "- Do NOT ask the user follow-up questions.\n"
        )
        user_chunks = []  # NEW
        for msg in messages:  # NEW
            if msg.get("role", "user") == "user":  # NEW
                user_chunks.append(str(msg.get("content", "")).strip())  # NEW
        question = "\n\n".join(user_chunks).strip()  # NEW
        parts.append(f"Question:\n{question}\n\nAnswer:")  # NEW
        return "\n".join(parts)  # NEW

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
                temp = max(0.0, min(0.7, float(self.temperature)))  # NEW: slightly lower temperature for stability
                do_sample = temp > 0.0
                top_p = min(self.top_p, 0.9)  # NEW: tighten nucleus sampling
                top_k = 0 if not do_sample else self.top_k  # NEW: deterministic when temp=0
                stop_criteria = StoppingCriteriaList([  # NEW: stop if model drifts into dialog markers
                    StopOnSeqs(self.tokenizer, ["\nHuman:", "\nUser:", "\nAssistant:\nHuman:", "\nAssistant:\nUser:"])
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
