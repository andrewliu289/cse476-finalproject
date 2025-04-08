from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import torch
import os

# Change this to switch between base and adapter model
USE_ADAPTER = os.getenv("USE_ADAPTER", "false").lower() == "true"

base_model_id = "meta-llama/Llama-3.2-3B"
adapter_path = "VishnuT/llama3-gsm8k-qlora-adapter"

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    adapter_path if USE_ADAPTER else base_model_id,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

print(f"Loading {'adapter' if USE_ADAPTER else 'base'} model")
# Runs with quantization
if USE_ADAPTER:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

model.eval()
print("Model loaded")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)