# data_utils.py
import random
import numpy as np
from datasets import load_dataset, Dataset

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
    """
    Transforms an MMLU training example into a prompt for fine-tuning.
    The prompt is: "Question: {question} Answer: {correct_choice}"
    Returns a tokenized output with keys such as 'input_ids', 'attention_mask',
    a computed 'token_length', and a 'labels' key (set equal to input_ids).
    """
    try:
        correct_choice = example["choices"][example["answer"]]
        prompt = f"Question: {example['question']} Answer: {correct_choice}"
        tokenized = tokenizer(prompt, truncation=True, max_length=512)
        tokenized["token_length"] = len(tokenized["input_ids"])
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    except Exception as e:
        print(f"Error tokenizing example: {e}. Returning default empty tokenization.")
        return {"input_ids": [], "attention_mask": [], "token_length": 0, "labels": []}

def create_subsample_groups(dataset, target_total_tokens, strategy='balanced', seed=42):
    """
    Given a tokenized dataset (with a "token_length" field), creates a subsample whose cumulative token_length
    does not exceed target_total_tokens.

    Supports both Hugging Face Datasets and plain Python lists.

    Strategies:
      - 'few_long': Prefer examples with high token_length.
      - 'many_short': Prefer examples with low token_length.
      - 'balanced': Prefer examples whose token_length is near the median.
    
    Returns:
      A list of examples (dictionaries) meeting the token budget.
    """
    # Shuffle the dataset.
    if hasattr(dataset, "shuffle"):
        # Hugging Face Dataset: use its shuffle method.
        dataset = dataset.shuffle(seed=seed)
    else:
        # Plain list.
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
    else:  # balanced strategy
        if isinstance(dataset, list):
            lengths = [ex["token_length"] for ex in dataset]
            median_length = np.median(lengths)
            sorted_dataset = sorted(dataset, key=lambda ex: abs(ex["token_length"] - median_length))
        else:
            # For a Hugging Face Dataset, add a new column "abs_diff".
            lengths = dataset["token_length"]
            median_length = np.median(lengths)
            # Add the column:
            dataset = dataset.map(lambda ex: {"abs_diff": abs(ex["token_length"] - median_length)})
            # Sort by the new column:
            sorted_dataset = dataset.sort("abs_diff")
    
    # Build the subsample until the cumulative token_length is as high as possible without exceeding the target.
    selected = []
    total = 0
    # For Hugging Face Dataset, convert to a list for iteration.
    if not isinstance(sorted_dataset, list):
        sorted_dataset = list(sorted_dataset)
    for ex in sorted_dataset:
        if total + ex["token_length"] <= target_total_tokens:
            selected.append(ex)
            total += ex["token_length"]
        else:
            break
    return selected
