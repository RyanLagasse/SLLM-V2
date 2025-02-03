# evaluate.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score

def compute_accuracy(model, eval_dataset, tokenizer):
    """
    Computes accuracy for MMLU-style evaluation examples.
    For each example, builds a prompt using the "question", "choices", and "answer" fields,
    then selects the candidate with the lowest loss.
    Returns the accuracy (a float between 0 and 1).
    """
    model.eval()
    predictions = []
    ground_truths = []
    for example in eval_dataset:
        if "question" in example:
            question = example["question"]
            choices = example["choices"]
            correct_answer = example["answer"]
            best_choice = None
            best_score = float("inf")
            for i, choice in enumerate(choices):
                prompt = f"Question: {question} Answer: {choice}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                if loss < best_score:
                    best_score = loss
                    best_choice = i
            predictions.append(best_choice)
            ground_truths.append(correct_answer)
    if len(ground_truths) == 0:
        return 0.0
    acc = accuracy_score(ground_truths, predictions)
    return acc

def compute_brimi_metrics(model, eval_dataset, tokenizer):
    """
    Computes a suite of metrics (accuracy, precision, recall, F1, F2, AUC) for BRIMI-style evaluation examples.
    
    Assumes each evaluation example has:
      - "text": the text to classify.
      - "classification": a string, either "True" or "False" (or lowercased variants).
    
    The function constructs a prompt for each candidate using:
      "Does this text exhibit gender bias? {text} Answer: {candidate}"
    with candidates ["False", "True"].
    
    For each example, it computes the loss for each candidate. Then it converts the losses
    to a probability for class 1 using a softmax over the negative losses.
    
    Finally, it computes and returns a dictionary of metrics.
    """
    model.eval()
    gt_labels = []
    pred_labels = []
    pred_probs = []  # probability for positive class (class 1)
    
    for example in eval_dataset:
        if "text" not in example:
            continue
        text = example["text"]
        # Define candidates for BRIMI: for instance, "False" and "True"
        candidates = ["False", "True"]
        # Map ground truth: assume classification is a string; convert to lowercase and strip.
        gt_str = example["classification"].strip().lower()
        gt = 0 if gt_str == "false" else 1
        gt_labels.append(gt)
        
        losses = []
        scores = []  # negative losses
        for candidate in candidates:
            prompt = f"Does this text exhibit gender bias? {text} Answer: {candidate}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            losses.append(loss)
            scores.append(-loss)  # lower loss means higher score
        
        # Compute softmax probabilities for the two classes.
        scores_np = np.array(scores)
        exp_scores = np.exp(scores_np - np.max(scores_np))  # subtract max for stability
        probs = exp_scores / np.sum(exp_scores)
        # Predicted class is the one with the highest probability.
        pred = int(np.argmax(probs))
        pred_labels.append(pred)
        # Store probability for class 1 (i.e., "True")
        pred_probs.append(probs[1])
    
    # Compute metrics using scikit-learn
    acc = accuracy_score(gt_labels, pred_labels)
    prec = precision_score(gt_labels, pred_labels, zero_division=0)
    rec = recall_score(gt_labels, pred_labels, zero_division=0)
    f1 = f1_score(gt_labels, pred_labels, zero_division=0)
    f2 = fbeta_score(gt_labels, pred_labels, beta=2, zero_division=0)
    try:
        auc = roc_auc_score(gt_labels, pred_probs)
    except ValueError:
        auc = float('nan')
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f2": f2,
        "auc": auc,
    }
    return metrics
