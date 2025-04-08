import collections
import json
import math
import re
import statistics
import time
from typing import List
import nltk
import numpy as np
import requests
import tracemalloc
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm


URL = "http://localhost:8000"

for i in ["punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{i}")
    except LookupError:
        nltk.download(i)


def getNum(text: str):
    # Return the last number found in the text
    if text is None:
        return None
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return float(matches[-1]) if matches else None


def mathError(correct: float, pred: float | None, raw_pred: str | None):
    if raw_pred is None or raw_pred.strip() == "":
        return "no_answer"
    if pred is None:
        return "parse_error"
    if correct is None:
        return "correct_missing"
    if abs(correct - pred) < 1e-3:
        return "correct"
    if re.search(r"[+\-*/^=]", raw_pred):
        return "arithmetic"
    return "logic"

# Math evaluation
def mathEval(path: str, url: str = URL):
    data = json.load(open(path))
    b = collections.Counter()

    for sample in tqdm(data, desc="[math]"):
        q = sample["question"]
        correct = getNum(sample["answer"])
        questions = [q]

        for q in questions:
            r = requests.post(f"{url}/model", json={"prompt": q}).json()
            resp = r.get("response", "")
            pred = getNum(resp)
            error = mathError(correct, pred, resp)
            b[error] += 1
            b["total"] += 1
            if pred is not None:
                b["answered"] += 1

    mathStats(b)
    return b

def mathStats(b: collections.Counter):
    total = b["total"] or 1
    print("\nMath Evaluation\n")
    print(f"Accuracy: {b['correct'] / total:.2%}")
    print(f"Answer Rate: {b['answered'] / total:.2%}")
    for k in ("arithmetic", "logic", "parse_error", "no_answer"):
        if b[k]:
            print(f"{k}: {b[k] / total:.2%}")

# Text generation evaluation
def textEval(path: str, url: str = URL):
    data = json.load(open(path))

    bleu, rouge1, rouge2, rougel, meteorVals, bertVals = [], [], [], [], [], []
    flags = collections.Counter()

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    refs: List[str] = []
    hyps: List[str] = []

    for sample in tqdm(data, desc="[text]"):
        ref = sample["reference"]
        inp = sample["input_text"]
        hyp = requests.post(f"{url}/model", json={"prompt": inp}).json().get("response", "")

        refs.append(ref)
        hyps.append(hyp)

        ref_tok = nltk.word_tokenize(ref.lower())
        hyp_tok = nltk.word_tokenize(hyp.lower())

        bleu.append(sentence_bleu([ref_tok], hyp_tok))
        meteorVals.append(meteor_score([ref], hyp))
        rs = rouge.score(ref, hyp)
        rouge1.append(rs["rouge1"].recall)
        rouge2.append(rs["rouge2"].recall)
        rougel.append(rs["rougeL"].recall)

        # Quick heuristics
        if hallucination(hyp):
            flags["hallucination"] += 1
        if repetition(hyp):
            flags["repetition"] += 1

    _, _, f1 = bert_score(hyps, refs, lang="en", rescale_with_baseline=True)
    bertVals = f1.tolist()

    textStats(bleu, rouge1, rouge2, rougel, meteorVals, bertVals, flags, len(data))
    return bleu, rouge1, rouge2, rougel, meteorVals, bertVals


def textStats(bleu, r1, r2, rl, meteorVals, bertVals, flags, n):
    def avg(x):
        return sum(x) / len(x) if x else 0.0

    print("\nText generation")
    print(f"BLEU: {avg(bleu):.3f}")
    print(f"ROUGE: R1 {avg(r1):.3f} | R2 {avg(r2):.3f} | RL {avg(rl):.3f}")
    print(f"METEOR: {avg(meteorVals):.3f}")
    print(f"BERT‑F1: {avg(bertVals):.3f}")
    for k, v in flags.items():
        print(f"{k.capitalize()}: {v / n:.2%}")
    print()

def hallucination(hyp: str) -> bool:
    return bool(re.search(r"\d{4}", hyp))

def repetition(hyp: str) -> bool:
    toks = hyp.split()
    return len(toks) != len(set(toks))

# Loss/perplexity
def lossPerplexityEval(path: str, url: str = URL):
    texts = json.load(open(path))
    losses, lengths = [], []

    for t in tqdm(texts, desc="[loss]"):
        resp = requests.post(f"{url}/compute_loss", json={"text": t}).json()
        losses.append(resp["loss"])
        lengths.append(resp["num_tokens"])

    totalLoss = sum(lengths) or 1
    total = 0
    for loss, num in zip(losses, lengths):
        total += loss * num

    avgLoss = total / totalLoss
    ppl = math.exp(avgLoss)

    print("\nLoss/Perplexity Evaluation\n")
    print(f"Total samples: {len(texts)}")
    print(f"Total tokens: {totalLoss}")
    print(f"Average loss: {avgLoss:.4f}")
    print(f"Perplexity: {ppl:.2f}")

    worst = np.argsort(losses)[-10:][::-1]
    print("\nWorst 10 sentences by loss:")
    for i in worst:
        print(f"[{losses[i]:.2f}] {texts[i][:120]}…")

    return avgLoss, ppl, losses

# Efficiency measurement
def measureEfficiency(prompt: str, url: str = URL, runs: int = 5):
    latencies = []
    for i in range(runs):
        t0 = time.perf_counter()
        _ = requests.post(f"{url}/model", json={"prompt": prompt}).json()
        latencies.append(time.perf_counter() - t0)

    print("\nEfficiency\n")
    print(f"Mean Latency: {statistics.mean(latencies)*1000:.1f} ms")
    return latencies

def main():
    while True:
        print("\nPick an Evaluation:")
        print("1. Math")
        print("2. Text")
        print("3. Loss")
        print("4. Efficiency")
        print("5. All of the Above")
        print("6. Exit")

        choice = input("Enter 1–6: ").strip()

        if choice == "1":
            mathEval("math.json")
        elif choice == "2":
            textEval("text.json")
        elif choice == "3":
            lossPerplexityEval("corpus.json")
        elif choice == "4":
            prompt = input("Enter prompt for efficiency test: ").strip()
            measureEfficiency(prompt)
        elif choice == "5":
            prompt = input("Enter prompt for efficiency test: ").strip()
            mathEval("math.json")
            textEval("text.json")
            lossPerplexityEval("corpus.json")
            measureEfficiency(prompt)
        elif choice == "6":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
