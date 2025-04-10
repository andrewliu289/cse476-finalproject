from fastapi import FastAPI
from model import generator, model, tokenizer
from pydantic import BaseModel
from typing import List
import torch

app = FastAPI()

class PromptRequest(BaseModel):
    prompts: List[str]

class LossRequest(BaseModel):
    texts: List[str]

@app.post("/model")
async def generate_batch(request: PromptRequest):
    prompts = request.prompts

    results = generator(prompts)

    responses = []
    for prompt, result in zip(prompts, results):
        full = result[0]["generated_text"]
        trimmed = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
        responses.append(trimmed)

    return {"responses": responses}


@app.post("/compute_loss")
async def compute_loss(req: LossRequest):
    losses = []
    num_tokens = []

    for text in req.texts:
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            losses.append(float(outputs.loss.item()))
            num_tokens.append(int(input_ids.size(1)))

    return {
        "losses": losses,
        "num_tokens": num_tokens
    }
