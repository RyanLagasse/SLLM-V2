# evaluate.py
import torch

def compute_accuracy(model, eval_dataset, tokenizer):
    """
    Computes accuracy on a multiple-choice task by ranking candidate answers via their log-likelihood loss.
    For each example, constructs a prompt for each candidate answer and selects the one with the lowest loss.
    """
    model.eval()
    correct, total = 0, 0
    for example in eval_dataset:
        question = example["question"]
        choices = example["choices"]
        correct_answer = example["answer"]
        
        best_choice = None
        best_score = float("inf")
        
        for i, choice in enumerate(choices):
            prompt = f"Question: {question} Answer: {choice}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            if loss < best_score:
                best_score = loss
                best_choice = i
        
        if best_choice == correct_answer:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy
