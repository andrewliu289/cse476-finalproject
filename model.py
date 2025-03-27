from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model = "meta-llama/Llama-3.2-3B"

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model)

print("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    model,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
