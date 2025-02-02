# main.py
import os
import csv
import time
import random
import numpy as np
from transformers import AutoTokenizer, TrainingArguments
from data_utils import load_mmlu_dataset, tokenize_training_example, create_subsample_groups, set_seed
from train import fine_tune_model

# -----------------------------
# Configuration Parameters
# -----------------------------
# MMLU configuration name (change this one value to pick a different subset)
MMLU_CONFIG = "human_aging"  # e.g., "human_aging", "college_biology", etc.

# Model to use
MODEL_LIST = [
    # "HuggingFaceTB/SmolLM-135M-Instruct",
    # "HuggingFaceTB/SmolLM-360M-Instruct",
    # "Qwen/Qwen2.5-0.5B-Instruct",
    # "tiiuae/Falcon3-1B-Instruct",
    "tiiuae/Falcon3-3B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "tiiuae/Falcon3-10B-Instruct",
]

# Use the first model for orchestration (you can later loop over models)
TEST_MODEL_NAME = MODEL_LIST[0]

# Random seeds to ensure reproducibility
RANDOM_SEEDS = [42]

# Target fractions for total token budget.
# For each dataset, we'll compute total tokens and then set target_tokens = fraction * (total tokens)
TARGET_TOKEN_FRACTIONS = [0.33, 0.66, 1.0]

# Subset sizes as fractions of the total number of training examples.
# We want to compare performance when using 25%, 50%, 75%, and 100% of the training examples.
SUBSET_FRACTIONS = [0.25, 0.50, 0.75, 1.0]

# Subsampling strategies to test.
SUBSAMPLING_STRATEGIES = ["few_long", "many_short", "balanced"]

# Output CSV file for logging results.
CSV_FILE = "experiment_results.csv"

# -----------------------------
# Helper Function: Compute total tokens and number of examples
# -----------------------------
def compute_dataset_metrics(tokenized_dataset):
    """
    Given a tokenized dataset (which can be a Hugging Face Dataset or a list),
    compute the total number of tokens (by summing 'token_length') and
    the total number of examples.
    """
    # If the dataset is a Hugging Face Dataset, convert the token_length column to a numpy array.
    if hasattr(tokenized_dataset, "column"):
        lengths = np.array(tokenized_dataset.column("token_length"))
        total_tokens = int(np.sum(lengths))
        num_examples = len(tokenized_dataset)
    else:
        # Otherwise, assume it's a list.
        lengths = [ex["token_length"] for ex in tokenized_dataset]
        total_tokens = int(sum(lengths))
        num_examples = len(tokenized_dataset)
    return total_tokens, num_examples

# -----------------------------
# Main Experimental Loop
# -----------------------------
def run_experiments():
    # Load the raw MMLU dataset using the specified configuration.
    raw_train, eval_dataset = load_mmlu_dataset(MMLU_CONFIG)
    print(f"Loaded MMLU subset '{MMLU_CONFIG}': {len(raw_train)} training examples, {len(eval_dataset)} evaluation examples.")
    
    # Initialize tokenizer from the test model.
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
    
    # Tokenize the training dataset.
    # We remove all original columns so that the tokenized version has only the fields produced by tokenize_training_example.
    tokenized_train = raw_train.map(
        lambda ex: tokenize_training_example(ex, tokenizer),
        remove_columns=raw_train.column_names,
        batched=False
    )
    
    # Optionally, filter out any examples that did not tokenize properly (missing 'labels').
    tokenized_train = tokenized_train.filter(lambda ex: "labels" in ex and len(ex["labels"]) > 0)
    
    # Compute overall metrics for the tokenized training dataset.
    total_tokens_available, total_examples = compute_dataset_metrics(tokenized_train)
    print(f"Total tokens in training data: {total_tokens_available}, Total examples: {total_examples}")
    
    # Determine target tokens based on fractions of the total available.
    target_tokens_list = [int(frac * total_tokens_available) for frac in TARGET_TOKEN_FRACTIONS]
    # Determine subset sizes (number of examples) based on fractions of the total examples.
    subset_sizes = sorted(list(set([max(1, int(frac * total_examples)) for frac in SUBSET_FRACTIONS])))
    
    print(f"Target tokens (by fraction): {target_tokens_list}")
    print(f"Subset sizes (number of examples): {subset_sizes}")
    
    # Open CSV file for logging.
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name", "tuning_method", "subset_strategy", "subset_size",
            "target_tokens", "random_seed", "avg_token_length", "total_tokens",
            "evaluation_dataset", "accuracy", "training_time"
        ])
        
        # Iterate over models (if you have more than one).
        for model_name in MODEL_LIST:
            # Loop over random seeds.
            for seed in RANDOM_SEEDS:
                set_seed(seed)
                # For each target token budget (as an absolute number).
                for target_tokens in target_tokens_list:
                    # For each subsampling strategy.
                    for subset_strategy in SUBSAMPLING_STRATEGIES:
                        # Use our custom subsampling function to select a subset from the tokenized training data.
                        subsample = create_subsample_groups(tokenized_train, target_tokens, strategy=subset_strategy, seed=seed)
                        # For each subset size (number of examples) we want to test.
                        for subset_size in subset_sizes:
                            # If subset_size is larger than the available subsample, use the full subsample.
                            if subset_size > len(subsample):
                                current_subset = subsample
                            else:
                                current_subset = subsample[:subset_size]
                            # Compute average token length and total tokens in the current subset.
                            if len(current_subset) > 0:
                                avg_token_length = np.mean([ex["token_length"] for ex in current_subset])
                                total_tokens_used = int(sum(ex["token_length"] for ex in current_subset))
                            else:
                                avg_token_length = 0
                                total_tokens_used = 0
                            
                            # Set up output directory for this experiment.
                            output_dir = f"./results/{model_name}_{subset_strategy}_{subset_size}_{target_tokens}_seed{seed}"
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # Create TrainingArguments.
                            # Note: We disable internal evaluation (evaluation_strategy="no") because our eval_dataset is raw.
                            training_args = TrainingArguments(
                                output_dir=output_dir,
                                num_train_epochs=1,  # For quick experiments; adjust as needed.
                                per_device_train_batch_size=8,
                                per_device_eval_batch_size=8,
                                evaluation_strategy="no",
                                save_strategy="no",
                                logging_steps=50,
                                fp16=True,
                            )
                            
                            print(f"Running experiment: model={model_name}, seed={seed}, "
                                  f"target_tokens={target_tokens}, strategy={subset_strategy}, "
                                  f"subset_size={len(current_subset)} (avg tokens {avg_token_length:.1f}, total {total_tokens_used})")
                            
                            # Run fine-tuning (or zero-shot if subset_size==0).
                            # Here, if current_subset is empty, we treat it as zero-shot.
                            tuning_method = "0-shot" if len(current_subset) == 0 else "FMT"
                            acc = fine_tune_model(model_name, current_subset if len(current_subset) > 0 else None, eval_dataset, output_dir, training_args)
                            training_time = 0  # Replace with actual timing if desired.
                            
                            # Log results.
                            writer.writerow([
                                model_name,
                                tuning_method,
                                subset_strategy,
                                len(current_subset),
                                target_tokens,
                                seed,
                                avg_token_length,
                                total_tokens_used,
                                f"MMLU:{MMLU_CONFIG}",
                                acc,
                                training_time,
                            ])
                            f.flush()
    
    print(f"Experiments completed. Results saved to {CSV_FILE}")

if __name__ == "__main__":
    run_experiments()
