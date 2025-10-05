import time
from typing import Any, Optional, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..types import MessageList, SamplerBase, SamplerResponse

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
            parts: List[str] = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"System: {content}\n")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}\n")
                else:
                    parts.append(f"Human: {content}\n")
            parts.append("Assistant:")
            text = "".join(parts)
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
                temp = max(0.0, min(1.5, float(self.temperature)))
                do_sample = temp > 0.0
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=temp,
                        do_sample=do_sample,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                input_len = inputs["input_ids"].shape[1]
                response_tokens = outputs[0][input_len:]
                content = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                if not content:
                    content = "[Empty response]"
                usage = {
                    "prompt_tokens": int(input_len),
                    "completion_tokens": int(response_tokens.shape[0]),
                    "total_tokens": int(input_len + response_tokens.shape[0]),
                }
                return SamplerResponse(
                    response_text=content,
                    response_metadata={
                        "model": self.model_id,
                        "device": str(self._device_str),
                        "usage": usage,
                    },
                    actual_queried_message_list=effective_messages,
                )
            except torch.cuda.OutOfMemoryError as e:
                print("CUDA OOM encountered; clearing cache and retrying once.")
                torch.cuda.empty_cache()
                if trial == 0:
                    trial += 1
                    continue
                else:
                    return SamplerResponse(
                        response_text=f"Error: {str(e)}",
                        response_metadata={"model": self.model_id, "error": "cuda_oom"},
                        actual_queried_message_list=effective_messages,
                    )
            except Exception as e:
                backoff = 2 ** trial
                print(f"Exception during generation; retry {trial} after {backoff} sec:", e)
                if trial >= 3:
                    return SamplerResponse(
                        response_text=f"Error: {str(e)}",
                        response_metadata={"model": self.model_id, "error": str(e)},
                        actual_queried_message_list=effective_messages,
                    )
                time.sleep(backoff)
                trial += 1
