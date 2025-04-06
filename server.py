from fastapi import FastAPI
from model import generator, model, tokenizer
from pydantic import BaseModel
import torch

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

class LossRequest(BaseModel):
    text: str

@app.post("/model")
async def generate(request: PromptRequest):
    prompt = request.prompt

    # Generate text
    result = generator(prompt)[0]["generated_text"]

    response = result[len(prompt):].strip() if result.startswith(prompt) else result.strip()
    return {"response": response}

@app.post("/compute_loss")
async def compute_loss(req: LossRequest):
    inputs = tokenizer(req.text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    num_tokens = input_ids.size(1)
    return {
        "loss": float(loss.item()),
        "num_tokens": int(num_tokens)
    }
