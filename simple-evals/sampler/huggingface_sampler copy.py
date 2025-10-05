import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any

from ..types import MessageList, SamplerBase, SamplerResponse


class HuggingFaceSampler(SamplerBase):
    """
    Sample from gpt-oss-20b on GPU
    """

    def __init__(
        self,
        model_choice: str = "gpt-oss-20b",
        system_message: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        device: str = "auto",
    ):
        self.model_choice = model_choice
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_gpt_neo()

    def _load_gpt_neo(self):
        """Load gpt-oss-20b optimized for GPU"""
        model_id = "openai/gpt-oss-20b"
        print(f"Loading gpt-oss-20bfrom {model_id}...")
        
        # Check GPU availability
        if torch.cuda.is_available():
            device = "cuda:0"
            torch_dtype = torch.float16  # Use half precision for GPU
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("CUDA not available, using CPU")
        #device = "cpu"
        #torch_dtype = torch.float32
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        
        # Move to device if not using device_map
        if not device.startswith("cuda"):
            model = model.to(device)
        
        print(f"Model loaded successfully on {device}")
        return model, tokenizer

    def _format_messages(self, message_list: MessageList) -> str:
        """Convert message list to prompt"""
        if self.system_message:
            prompt = f"{self.system_message}\n\n"
        else:
            prompt = ""
            
        for message in message_list:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                prompt += f"Human: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
            elif role == "system":
                prompt += f"System: {content}\n"
                
        prompt += "Assistant:"
        return prompt

    def _pack_message(self, role: str, content: Any):
        """Pack message for compatibility"""
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        try:
            # Format prompt
            prompt = self._format_messages(message_list)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=800,  # Leave room for generation
                padding=False,
            )
            
            # Move to model device
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                # Ensure stable generation
                temp = max(0.1, min(1.5, self.temperature))
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=temp,
                    do_sample=temp > 0,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    #early_stopping=True,
                )
            
            # Extract generated text
            input_length = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            if not response_text:
                response_text = "[Empty response]"
            
            return SamplerResponse(
                response_text=response_text,
                response_metadata={
                    "model": "gpt-oss-20b",
                    "device": str(model_device),
                    "tokens_generated": len(response_tokens)
                },
                actual_queried_message_list=message_list,
            )
            
        except Exception as e:
            print(f"Generation error: {e}")
            return SamplerResponse(
                response_text=f"Error: {str(e)}",
                response_metadata={"model": "gpt-oss-20b", "error": str(e)},
                actual_queried_message_list=message_list,
            )