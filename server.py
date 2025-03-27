from fastapi import FastAPI
from model import generator
from pydantic import BaseModel
import json
import random

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

# Use Dev Data toggle
# Set True when you want to use dev_data for few-shot prompting
# Set False when you 
useData = False
# Batch size so you can see progress
batchSize = 100

data = []
if useData:
    with open("dev_data.json", "r") as f:
        rawData = json.load(f)
        total = len(rawData)
        print(f"Loading {total}")

        for i in range(0, total, batchSize):
            batch = rawData[i:i + batchSize]
            data.extend(batch)
            num = (i // batchSize) + 1
            print(f"Batch {num} ({len(data)}/{total})")

    print(f"Finished loading")

def formatPrompt(input: str, num: int = 5) -> str:
    if not useData or not data:
        return input
    
    examples = random.sample(data, num)

    formatted = ""
    for i in examples:
        formatted += f"<PROMPT>\n{i['question']}\n<RESPONSE>\n{i['answer']}\n\n"

    formatted += f"<PROMPT>\n{input}\n<RESPONSE>\n"
    return formatted

@app.post("/chat")
async def chat(request: PromptRequest):
    prompt = formatPrompt(request.prompt)
    result = generator(prompt)[0]["generated_text"]

    if useData:
        response = result[len(prompt):].strip()
    else:
        response = result.strip()

    return {"response": response}
