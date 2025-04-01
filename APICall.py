import requests
import re

url = "http://127.0.0.1:8000/chat"

def query_model(prompt: str) -> str:
    payload = {"prompt": prompt}
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["response"]
    except Exception as e:
        print("Error calling model")
        return ""
    
def arithmetic():
    prompts = [
        ("What is 2 + 2?", "4"),
        ("Calculate 17 minus 5.", "12"),
        ("What is 7 multiplied by 6?", "42"),
        ("What is 81 divided by 9?", "9"),
        ("What is 3 plus 14?", "17"),
    ]

    results = []
    for prompt, expected in prompts:
        answer = query_model(prompt).strip()
        # Extract the first word/number
        predicted = re.split(r"[\s,.;?!]+", answer)[0] if answer else ""
        correct = (predicted == expected)
        results.append((prompt, expected, predicted, correct))
    return results

def analogies():
    prompts = [
        ("King is to Queen as Man is to", "Woman"),
        ("Paris is to France as Tokyo is to", "Japan"),
        ("Dog is to Puppy as Cat is to", "Kitten"),
        ("Bird is to Fly as Fish is to", "Swim"),
    ]

    results = []
    for prompt, expected in prompts:
        answer = query_model(prompt).strip()
        # Extract the first word/number
        predicted = re.split(r"[\s,.;?!]+", answer)[0] if answer else ""
        correct = (predicted.lower() == expected.lower())
        results.append((prompt, expected, predicted, correct))
    return results

def multiple_choice():
    questions = [
        {
            "question": "King is to Queen as Man is to",
            "choices": ["A) Woman", "B) Boy", "C) Prince", "D) Girl"],
            "correct": "A) Woman"
        },
        {
            "question": "Paris is to France as Tokyo is to",
            "choices": ["A) Japan", "B) China", "C) Seoul", "D) Bangkok"],
            "correct": "A) Japan"
        }
    ]

    results = []
    for item in questions:
        q = item["question"]
        choices_str = "\n".join(item["choices"])
        prompt = f"{q}:\n{choices_str}\nWhich one is correct?"
        model_answer = query_model(prompt).strip()

        chosen = None
        for c in item["choices"]:
            label = c.split(")")[0] + ")"
            text = c.split(")")[1].strip()
            if label in model_answer or text in model_answer:
                chosen = c
                break

        correct = (chosen == item["correct"])
        results.append((prompt, item["correct"], chosen, correct))

    return results

# Evals
def main():
    print("Arithmetic Evaluation:")
    arithmetic_results = arithmetic()
    correct_count = sum(r[3] for r in arithmetic_results)
    print(f"Arithmetic Accuracy: {correct_count}/{len(arithmetic_results)} = {correct_count/len(arithmetic_results):.2%}")
    for r in arithmetic_results:
        prompt, expected, predicted, correct = r
        print(f"Q: {prompt} | Expected: {expected} | Got: {predicted} | {'Correct' if correct else 'Wrong'}")

    print("Analogy Evaluation:")
    analogy_results = analogies()
    correct_count = sum(r[3] for r in analogy_results)
    print(f"Analogy Accuracy: {correct_count}/{len(analogy_results)} = {correct_count/len(analogy_results):.2%}")
    for r in analogy_results:
        prompt, expected, predicted, correct = r
        print(f"Q: {prompt} | Expected: {expected} | Got: {predicted} | {'Correct' if correct else 'Wrong'}")

    print("Multiple Choice Evaluation:")
    mc_results = multiple_choice()
    correct_count = sum(r[3] for r in mc_results)
    print(f"Multiple-choice Accuracy: {correct_count}/{len(mc_results)} = {correct_count/len(mc_results):.2%}")
    for r in mc_results:
        prompt, expected, chosen, correct = r
        print(f"Q: {prompt}")
        print(f"A: {expected}")
        print(f"Model: {chosen}")
        print(f"{'Correct' if correct else 'Wrong'}")

if __name__ == "__main__":
    main()