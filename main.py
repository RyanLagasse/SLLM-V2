# main.py
import os
import csv
import time
import numpy as np
from transformers import AutoTokenizer, TrainingArguments
from data_utils import set_seed, load_mmlu_dataset, tokenize_training_example, create_subsample_groups
from train import fine_tune_model

def run_experiments():
    # Experiment settings.
    MODEL_LIST = [
        "HuggingFaceTB/SmolLM-135M-Instruct",
        "HuggingFaceTB/SmolLM-360M-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "tiiuae/Falcon3-1B-Instruct",
        "tiiuae/Falcon3-3B-Instruct",
        "tiiuae/Falcon3-7B-Instruct",
        "tiiuae/Falcon3-10B-Instruct",
    ]
    TRAIN_SUBSET_SIZES = [0, 50, 75, 100, 125]  # Number of training examples (0 = zero-shot)
    TARGET_TOTAL_TOKENS_LIST = [1000, 2000, 3000]      # Overall token budgets
    RANDOM_SEEDS = [42, 100, 2025]
    
    # MMLU dataset details.
    FINE_TUNE_DATASET_CONFIG = "anatomy"  # Using "cais/mmlu" with config "anatomy"
    
    # Load raw MMLU dataset (train and eval).
    raw_train, raw_eval = load_mmlu_dataset(FINE_TUNE_DATASET_CONFIG)
    
    # Use the first model's tokenizer for training tokenization.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LIST[0])
    
    # Tokenize training examples into prompts and remove original columns.
    tokenized_train = raw_train.map(
        lambda ex: tokenize_training_example(ex, tokenizer),
        remove_columns=raw_train.column_names,
        batched=False
    )

    for i, ex in enumerate(tokenized_train):
        if "labels" not in ex:
            print(f"Warning: Example {i} is missing 'labels':", ex)
    
    # We'll use the raw eval set for evaluation.
    eval_dataset = raw_eval

    # Set up CSV logging.
    csv_file = "experiment_results.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name", "tuning_method", "subset_strategy", "subset_size",
            "target_tokens", "random_seed", "avg_token_length", "total_tokens",
            "evaluation_dataset", "accuracy", "training_time"
        ])

        # Loop over experimental configurations.
        for model_name in MODEL_LIST:
            for seed in RANDOM_SEEDS:
                set_seed(seed)
                for target_total_tokens in TARGET_TOTAL_TOKENS_LIST:
                    for subset_strategy in ['few_long', 'many_short', 'balanced']:
                        subsample = create_subsample_groups(tokenized_train, target_total_tokens, strategy=subset_strategy, seed=seed)
                        for subset_size in TRAIN_SUBSET_SIZES:
                            if subset_size == 0:
                                # Zero-shot evaluation: Set evaluation strategy to "no" to bypass Trainer check.
                                output_dir = f"./results/{model_name}_0shot_{subset_strategy}_{target_total_tokens}_seed{seed}"
                                os.makedirs(output_dir, exist_ok=True)
                                training_args = TrainingArguments(
                                    output_dir=output_dir,
                                    num_train_epochs=0,
                                    per_device_eval_batch_size=8,
                                    evaluation_strategy="no",  # <-- Changed for zero-shot.
                                    save_strategy="no",
                                    logging_steps=10,
                                    fp16=True,
                                    remove_unused_columns=False,
                                )
                                start_time = time.time()
                                acc = fine_tune_model(model_name, None, eval_dataset, output_dir, training_args)
                                training_time = time.time() - start_time
                                writer.writerow([
                                    model_name, "0-shot", subset_strategy, subset_size,
                                    target_total_tokens, seed, None, 0,
                                    "MMLU:anatomy", acc, training_time
                                ])
                                f.flush()
                                continue

                            train_subset = subsample[:subset_size]
                            avg_token_length = np.mean([ex["token_length"] for ex in train_subset])
                            total_tokens = sum(ex["token_length"] for ex in train_subset)
                            output_dir = f"./results/{model_name}_{subset_strategy}_{subset_size}_{target_total_tokens}_seed{seed}"
                            os.makedirs(output_dir, exist_ok=True)
                            training_args = TrainingArguments(
                                output_dir=output_dir,
                                num_train_epochs=3,
                                per_device_train_batch_size=8,
                                per_device_eval_batch_size=8,
                                evaluation_strategy="no",
                                save_strategy="epoch",
                                logging_steps=50,
                                fp16=True,
                                remove_unused_columns=False,
                            )
                            start_time = time.time()
                            acc = fine_tune_model(model_name, train_subset, eval_dataset, output_dir, training_args)
                            training_time = time.time() - start_time
                            writer.writerow([
                                model_name, "FMT", subset_strategy, subset_size,
                                target_total_tokens, seed, avg_token_length, total_tokens,
                                "MMLU:anatomy", acc, training_time
                            ])
                            f.flush()

if __name__ == "__main__":
    run_experiments()
