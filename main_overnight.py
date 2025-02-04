# main.py
import os
import csv
import time
import random
import numpy as np
import torch
from transformers import AutoTokenizer, TrainingArguments
from data_utils import (
    set_seed,
    load_mmlu_dataset,
    tokenize_training_example,
    create_subsample_groups,
    compute_dataset_metrics,
)
from train import fine_tune_model

# -----------------------------
# Dataset Selection and Tokenization Functions for BRIMI
# -----------------------------
def load_brimi_dataset(folder="/opt/extra/avijit/projects/rlof/datasets/bricc_no_instructions", name="bricc_gender_20"):
    """
    Loads the BRIMI dataset from JSON files.
    Returns (train_dataset, eval_dataset).
    """
    from datasets import load_dataset
    train_dataset = load_dataset("json", data_files=f"{folder}/{name}_train.json", split="train")
    eval_dataset = load_dataset("json", data_files=f"{folder}/{name}_eval.json", split="train")
    return train_dataset, eval_dataset

def tokenize_training_example_brimi(example, tokenizer):
    """
    Transforms a BRIMI example into a prompt for fine-tuning.
    The prompt is formed from the 'text' field.
    Returns a tokenized output with keys: 'input_ids', 'attention_mask', 'token_length', and 'labels'.
    """
    try:
        # Reformat the text if needed.
        text = f"Text: {example['text'].strip()}"
        # For training, we include the classification in the prompt if desired.
        # Here, we simply use the text; the model is expected to learn the mapping implicitly.
        prompt = text  # Alternatively, you could append "\nClassification:" if that suits your training.
        tokenized = tokenizer(prompt, truncation=True, max_length=512)
        tokenized["token_length"] = len(tokenized["input_ids"])
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    except Exception as e:
        print(f"Error tokenizing BRIMI example: {e}. Returning empty tokenization.")
        return {"input_ids": [], "attention_mask": [], "token_length": 0, "labels": []}

# -----------------------------
# Evaluation Functions (for MMLU and BRIMI)
# -----------------------------
def compute_accuracy(model, eval_dataset, tokenizer):
    """
    For MMLU-style evaluation: uses 'question', 'choices', 'answer' fields.
    Returns a dictionary with key 'accuracy'.
    """
    model.eval()
    predictions = []
    ground_truths = []
    for example in eval_dataset:
        if "question" not in example:
            continue
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
        return {"accuracy": 0.0}
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(ground_truths, predictions)
    return {"accuracy": acc}

def compute_brimi_metrics(model, eval_dataset, tokenizer):
    """
    For BRIMI-style evaluation: assumes each example has 'text' and 'classification'.
    Constructs prompts of the form:
      "Does this text exhibit gender bias? {text} Answer: {candidate}"
    with candidates ["False", "True"]. Maps ground truth to 0 if "false", else 1.
    Returns a dictionary with keys: accuracy, precision, recall, f1, f2, auc.
    """
    model.eval()
    gt_labels = []
    pred_labels = []
    pred_probs = []  # probability for positive class
    for example in eval_dataset:
        if "text" not in example:
            continue
        text = example["text"]
        candidates = ["False", "True"]
        gt_str = example["classification"].strip().lower()
        gt = 0 if gt_str == "false" else 1
        gt_labels.append(gt)
        losses = []
        scores = []
        for i, candidate in enumerate(candidates):
            prompt = f"Does this text exhibit gender bias? {text} Answer: {candidate}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            losses.append(loss)
            scores.append(-loss)  # lower loss -> higher score
        scores_np = np.array(scores)
        exp_scores = np.exp(scores_np - np.max(scores_np))
        probs = exp_scores / np.sum(exp_scores)
        pred = int(np.argmax(probs))
        pred_labels.append(pred)
        pred_probs.append(probs[1])
    # Compute metrics using scikit-learn.
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score
    if len(gt_labels) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0, "auc": float('nan')}
    acc = accuracy_score(gt_labels, pred_labels)
    prec = precision_score(gt_labels, pred_labels, zero_division=0)
    rec = recall_score(gt_labels, pred_labels, zero_division=0)
    f1 = f1_score(gt_labels, pred_labels, zero_division=0)
    f2 = fbeta_score(gt_labels, pred_labels, beta=2, zero_division=0)
    try:
        auc = roc_auc_score(gt_labels, pred_probs)
    except Exception:
        auc = float('nan')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "f2": f2, "auc": auc}

# -----------------------------
# Configuration for Experiments
# -----------------------------
# Set DATASET_TYPE: "MMLU" or "BRIMI"
DATASET_TYPE = "MMLU"  # Change to "MMLU" to run on an MMLU subset.
MMLU_CONFIG = "moral_scenarios"  # used if DATASET_TYPE == "MMLU"
# MMLU_CONFIG = "professional_law"  # used if DATASET_TYPE == "MMLU"


# Model list (you can add or remove models)
MODEL_LIST = [
    "HuggingFaceTB/SmolLM-135M-Instruct",
    "HuggingFaceTB/SmolLM-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "tiiuae/Falcon3-1B-Instruct",
    "tiiuae/Falcon3-3B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "tiiuae/Falcon3-10B-Instruct",
]
TEST_MODEL_NAME = MODEL_LIST[0]

RANDOM_SEEDS = [42]

TARGET_TOKEN_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
SUBSET_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
SUBSAMPLING_STRATEGIES = ["few_long", "many_short", "balanced"]

CSV_FILE = "experiment_results_moral.csv"

# -----------------------------
# Main Experimental Loop
# -----------------------------
def run_experiments():
    # Load dataset based on DATASET_TYPE.
    if DATASET_TYPE == "MMLU":
        raw_train, eval_dataset = load_mmlu_dataset(MMLU_CONFIG)
        eval_dataset_name = f"MMLU:{MMLU_CONFIG}"
        tokenize_fn = tokenize_training_example
    elif DATASET_TYPE == "BRIMI":
        from datasets import load_dataset  # In case not imported in data_utils
        # Use the provided loader for BRIMI.
        def load_brimi_dataset(folder="/opt/extra/avijit/projects/rlof/datasets/bricc_no_instructions", name="bricc_gender_20"):
            train_dataset = load_dataset("json", data_files=f"{folder}/{name}_train.json", split="train")
            eval_dataset = load_dataset("json", data_files=f"{folder}/{name}_eval.json", split="train")
            return train_dataset, eval_dataset
        raw_train, eval_dataset = load_brimi_dataset()
        eval_dataset_name = "BRIMI"
        tokenize_fn = tokenize_training_example_brimi
    else:
        raise ValueError("Unknown DATASET_TYPE. Choose 'MMLU' or 'BRIMI'.")

    print(f"Loaded {DATASET_TYPE} dataset: {len(raw_train)} training examples, {len(eval_dataset)} evaluation examples.")

    # Initialize tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)

    # Tokenize training data.
    tokenized_train = raw_train.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        remove_columns=raw_train.column_names,
        batched=False
    )
    tokenized_train = tokenized_train.filter(lambda ex: "labels" in ex and len(ex["labels"]) > 0)

    total_tokens_available, total_examples = compute_dataset_metrics(tokenized_train)
    print(f"Total tokens in training data: {total_tokens_available}, Total examples: {total_examples}")

    target_tokens_list = [int(frac * total_tokens_available) for frac in TARGET_TOKEN_FRACTIONS]
    subset_sizes = sorted(list(set([max(1, int(frac * total_examples)) for frac in SUBSET_FRACTIONS])))

    print(f"Target tokens (absolute): {target_tokens_list}")
    print(f"Subset sizes (number of examples): {subset_sizes}")

    # Open CSV for logging.
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name", "tuning_method", "subset_strategy", "subset_size",
            "target_tokens", "random_seed", "avg_token_length", "total_tokens",
            "evaluation_dataset", "metrics", "training_time"
        ])

        for model_name in MODEL_LIST:
            for seed in RANDOM_SEEDS:
                set_seed(seed)
                for target_tokens in target_tokens_list:
                    for subset_strategy in SUBSAMPLING_STRATEGIES:
                        subsample = create_subsample_groups(tokenized_train, target_tokens, strategy=subset_strategy, seed=seed)
                        for subset_size in subset_sizes:
                            if subset_size > len(subsample):
                                current_subset = subsample
                            else:
                                current_subset = subsample[:subset_size]
                            if len(current_subset) > 0:
                                avg_token_length = np.mean([ex["token_length"] for ex in current_subset])
                                total_tokens_used = int(sum(ex["token_length"] for ex in current_subset))
                            else:
                                avg_token_length = 0
                                total_tokens_used = 0

                            output_dir = f"./results/{model_name}_{subset_strategy}_{subset_size}_{target_tokens}_seed{seed}"
                            os.makedirs(output_dir, exist_ok=True)

                            # Use TrainingArguments with evaluation disabled.
                            training_args = TrainingArguments(
                                output_dir=output_dir,
                                num_train_epochs=1,
                                per_device_train_batch_size=8,
                                per_device_eval_batch_size=8,
                                evaluation_strategy="no",
                                save_strategy="no",
                                logging_steps=50,
                                fp16=True,
                            )

                            print(f"Running: model={model_name}, seed={seed}, target_tokens={target_tokens}, "
                                  f"strategy={subset_strategy}, subset_size={len(current_subset)} "
                                  f"(avg tokens {avg_token_length:.1f}, total {total_tokens_used})")

                            tuning_method = "0-shot" if len(current_subset) == 0 else "FMT"
                            # For BRIMI, use eval_fn="brimi" so that the fine_tune_model function computes full metrics.
                            if DATASET_TYPE == "BRIMI":
                                metrics = fine_tune_model(model_name, current_subset if len(current_subset) > 0 else None,
                                                          eval_dataset, output_dir, training_args, eval_fn="brimi")
                            else:
                                metrics = fine_tune_model(model_name, current_subset if len(current_subset) > 0 else None,
                                                          eval_dataset, output_dir, training_args, eval_fn="mmlu")
                            
                            # Convert metrics dict to a string for logging.
                            metrics_str = ";".join([f"{k}:{v:.3f}" for k, v in metrics.items()])
                            training_time = 0  # Replace with actual timing if desired.
                            writer.writerow([
                                model_name,
                                tuning_method,
                                subset_strategy,
                                len(current_subset),
                                target_tokens,
                                seed,
                                avg_token_length,
                                total_tokens_used,
                                eval_dataset_name,
                                metrics_str,
                                training_time,
                            ])
                            f.flush()

    print(f"Experiments completed. Results saved to {CSV_FILE}")

if __name__ == "__main__":
    run_experiments()
