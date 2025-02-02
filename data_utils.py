# data_utils.py
import random
import numpy as np
from datasets import load_dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
def load_mmlu_dataset(config="anatomy"):
    """
    Loads the MMLU dataset (cais/mmlu) with the given configuration.
    If there is no train split, splits the "test" split into train and eval.
    Returns (train_dataset, eval_dataset).
    """
    dataset = load_dataset("cais/mmlu", config)
    if "train" in dataset:
        train_ds = dataset["train"]
        eval_ds = dataset["test"] if "test" in dataset else dataset["train"].train_test_split(test_size=0.2, seed=42)["test"]
    else:
        split = dataset["test"].train_test_split(test_size=0.2, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]
    return train_ds, eval_ds

def tokenize_training_example(example, tokenizer):
    try:
        correct_choice = example["choices"][example["answer"]]
        prompt = f"Question: {example['question']} Answer: {correct_choice}"
        tokenized = tokenizer(prompt, truncation=True, max_length=512)
        tokenized["token_length"] = len(tokenized["input_ids"])
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    except Exception as e:
        # Log the error and return a dictionary with default empty values.
        print(f"Error tokenizing example: {e}. Returning default empty tokenization.")
        return {"input_ids": [], "attention_mask": [], "token_length": 0, "labels": []}



def create_subsample_groups(dataset, target_total_tokens, strategy='balanced', seed=42):
    """
    Given a tokenized dataset (with a "token_length" field), creates a subsample whose total tokens
    do not exceed the target_total_tokens.
    
    Supports both Hugging Face Datasets (with .shuffle() and .sort() methods)
    and plain Python lists.
    
    Strategies:
      - 'few_long': Prefer examples with high token_length.
      - 'many_short': Prefer examples with low token_length.
      - 'balanced': Prefer examples whose token_length is near the median.
    
    Returns:
      A list of examples (dictionaries) whose cumulative token_length does not exceed target_total_tokens.
    """
    # Shuffle the dataset.
    if hasattr(dataset, "shuffle"):
        # For a Hugging Face Dataset, use its .shuffle() method.
        dataset = dataset.shuffle(seed=seed)
    else:
        # Otherwise, assume it's a plain list.
        random.seed(seed)
        dataset = dataset.copy()
        random.shuffle(dataset)
    
    # Sorting based on the strategy.
    if strategy == 'few_long':
        if isinstance(dataset, list):
            sorted_dataset = sorted(dataset, key=lambda ex: ex["token_length"], reverse=True)
        else:
            sorted_dataset = dataset.sort("token_length", reverse=True)
    elif strategy == 'many_short':
        if isinstance(dataset, list):
            sorted_dataset = sorted(dataset, key=lambda ex: ex["token_length"], reverse=False)
        else:
            sorted_dataset = dataset.sort("token_length", reverse=False)
    else:  # balanced strategy.
        if isinstance(dataset, list):
            lengths = [ex["token_length"] for ex in dataset]
            median_length = np.median(lengths)
            sorted_dataset = sorted(dataset, key=lambda ex: abs(ex["token_length"] - median_length))
        else:
            try:
                lengths = dataset["token_length"]
            except Exception:
                lengths = [ex["token_length"] for ex in dataset]
            median_length = np.median(lengths)
            sorted_dataset = dataset.sort(key=lambda ex: abs(ex["token_length"] - median_length))
    
    # Build the subsample: only add an example if doing so does not exceed target_total_tokens.
    selected = []
    total = 0
    for ex in sorted_dataset:
        if total + ex["token_length"] <= target_total_tokens:
            selected.append(ex)
            total += ex["token_length"]
        else:
            break
    return selected
