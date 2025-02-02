# test_pipeline.py
import os
import time
import random
import numpy as np
import torch

from data_utils import set_seed, tokenize_training_example, create_subsample_groups
from evaluate import compute_accuracy
from train import fine_tune_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

# -----------------------------
# Helper Functions and Globals
# -----------------------------
TEST_MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"  # Realistic small LLM for testing

def get_tokenizer():
    return AutoTokenizer.from_pretrained(TEST_MODEL_NAME)

def get_model():
    model = AutoModelForCausalLM.from_pretrained(TEST_MODEL_NAME)
    # Ensure the model has a device attribute.
    try:
        _ = model.device
    except AttributeError:
        # Set the device to CPU if not available.
        model.device = torch.device("cpu")
    return model

# Create a dummy example that mimics an MMLU example.
dummy_example = {
    "question": "What is the capital of France?",
    "subject": "geography",
    "choices": ["Paris", "Berlin", "Madrid", "Rome"],
    "answer": 0  # Correct answer is index 0 ("Paris")
}

# Create a dummy tokenized dataset (list of dictionaries) for subsampling tests.
# We'll simulate examples with various token_length values.
dummy_tokenized_dataset = []
for i in range(10):
    example = {
        "input_ids": [101] * (10 + i),  # Example token sequence (length 10 to 19)
        "attention_mask": [1] * (10 + i),
        "token_length": 10 + i,
    }
    dummy_tokenized_dataset.append(example)

# Dummy evaluation dataset: a list with a single raw example.
dummy_eval_dataset = [dummy_example]

# -----------------------------
# Test 1: Tokenization
# -----------------------------
def test_tokenize_training_example():
    tokenizer = get_tokenizer()
    tokenized = tokenize_training_example(dummy_example, tokenizer)
    # Check that tokenized output has expected keys.
    assert "input_ids" in tokenized, "Missing input_ids."
    assert "token_length" in tokenized, "Missing token_length."
    assert "labels" in tokenized, "Missing labels in tokenized output."
    # Ensure token_length matches length of input_ids.
    assert tokenized["token_length"] == len(tokenized["input_ids"]), "Mismatch in token_length."
    # Check that labels match input_ids.
    assert tokenized["labels"] == tokenized["input_ids"], "Labels do not match input_ids."
    print("test_tokenize_training_example passed.")


# -----------------------------
# Test 2: Subsample Groups
# -----------------------------
def test_create_subsample_groups():
    target = 50
    subsample = create_subsample_groups(dummy_tokenized_dataset, target, strategy="balanced", seed=42)
    total = sum(ex["token_length"] for ex in subsample)
    # The total should not exceed the target.
    assert total <= target, f"Subsample total tokens {total} exceed target {target}."
    print("test_create_subsample_groups passed.")

# -----------------------------
# Test 3: Compute Accuracy
# -----------------------------
def test_compute_accuracy():
    tokenizer = get_tokenizer()
    model = get_model()
    # For this test, we use the dummy_eval_dataset.
    # Since we are running the pretrained model in zero-shot mode,
    # accuracy is not expected to be 1.0; instead, we check that the returned value is a float.
    acc = compute_accuracy(model, dummy_eval_dataset, tokenizer)
    assert isinstance(acc, float), "Accuracy should be a float."
    assert 0.0 <= acc <= 1.0, "Accuracy should be between 0 and 1."
    print(f"test_compute_accuracy passed with accuracy = {acc:.3f}")

# -----------------------------
# Test 4: Fine-tune Model Zero-Shot
# -----------------------------
def test_fine_tune_model_zero_shot():
    output_dir = "./dummy_results_zero_shot"
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=0,  # Zero-shot evaluation
        per_device_eval_batch_size=1,
        evaluation_strategy="no",  # No evaluation during training
        save_strategy="no",
        logging_steps=1,
        fp16=False,
        remove_unused_columns=False,
    )
    acc = fine_tune_model(TEST_MODEL_NAME, None, dummy_eval_dataset, output_dir, training_args)
    assert isinstance(acc, float), "Zero-shot accuracy should be a float."
    print(f"test_fine_tune_model_zero_shot passed with accuracy = {acc:.3f}")

# -----------------------------
# Test 5: Fine-tune Model With Training Data
# -----------------------------
def test_fine_tune_model_with_training_data():
    tokenizer = get_tokenizer()
    # Create a minimal training dataset by tokenizing the dummy example.
    train_dataset = [tokenize_training_example(dummy_example, tokenizer)]
    output_dir = "./dummy_results_train"
    os.makedirs(output_dir, exist_ok=True)
    # Here we set evaluation_strategy="no" to prevent Trainer from trying to collate raw eval data.
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Minimal training
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="no",  # Disable internal evaluation
        save_strategy="no",
        logging_steps=1,
        fp16=False,
        # remove_unused_columns not used if causing issues.
    )
    acc = fine_tune_model(TEST_MODEL_NAME, train_dataset, dummy_eval_dataset, output_dir, training_args)
    assert isinstance(acc, float), "Accuracy with training should be a float."
    print(f"test_fine_tune_model_with_training_data passed with accuracy = {acc:.3f}")

def test_custom_data_collator():
    from data_collator import custom_data_collator
    tokenizer = get_tokenizer()
    # Create a mini-batch with two examples of different lengths.
    example1 = {
        "input_ids": [101, 102, 103],
        "attention_mask": [1, 1, 1],
        "labels": [101, 102, 103],
        "token_length": 3
    }
    example2 = {
        "input_ids": [101, 102],
        "attention_mask": [1, 1],
        "labels": [101, 102],
        "token_length": 2
    }
    batch = custom_data_collator([example1, example2], tokenizer)
    # Print out the batch for debugging.
    print("Collated batch keys:", batch.keys())
    print("input_ids shape:", batch["input_ids"].shape)
    print("labels shape:", batch["labels"].shape)
    # Check that batch has the keys we expect.
    assert "labels" in batch, "Batch is missing 'labels'"
    # Check that the padded shapes are equal.
    assert batch["input_ids"].shape[1] == batch["labels"].shape[1], "Padded sequence lengths do not match."
    # Ensure that there are two examples in the batch.
    assert batch["input_ids"].shape[0] == 2, "Batch size is not 2."
    print("test_custom_data_collator passed.")



def test_full_integration_on_real_dataset():
    """
    Integration test for the full pipeline using the real MMLU dataset.
    Loads a small subset of the real MMLU training and evaluation data,
    tokenizes the training examples, and runs a minimal training and evaluation loop.
    Asserts that the final computed accuracy is between 0 and 1.
    """
    from data_utils import load_mmlu_dataset, tokenize_training_example
    from transformers import TrainingArguments
    from train import fine_tune_model
    from transformers import AutoTokenizer

    # Load the real MMLU dataset with configuration "anatomy"
    train_ds, eval_ds = load_mmlu_dataset(config="anatomy")
    print(f"Real dataset loaded: train size = {len(train_ds)}, eval size = {len(eval_ds)}")
    
    # Use our test model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
    
    # Tokenize the training dataset.
    # We remove original columns so that only the tokenized fields remain.
    tokenized_train = train_ds.map(
        lambda ex: tokenize_training_example(ex, tokenizer),
        remove_columns=train_ds.column_names,
        batched=False
    )
    
    # For a quick test, select a small subset of the tokenized training data.
    # (Adjust the number if necessary.)
    if hasattr(tokenized_train, "select"):
        # Hugging Face Dataset supports select.
        train_subset = tokenized_train.select(range(min(10, len(tokenized_train))))
    else:
        # Otherwise assume it's a list.
        train_subset = tokenized_train[:10]
    
    output_dir = "./integration_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up training arguments.
    # We disable internal evaluation (evaluation_strategy="no") because our eval dataset is raw.
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Minimal training for integration testing
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="no",  # Disable internal evaluation to avoid collator issues on raw eval data
        save_strategy="no",
        logging_steps=1,
        fp16=False,
        # remove_unused_columns can be omitted if problematic
    )
    
    # Run fine-tuning on the real dataset subset.
    acc = fine_tune_model(TEST_MODEL_NAME, train_subset, eval_ds, output_dir, training_args)
    print(f"Integration test accuracy: {acc:.3f}")
    
    # Basic assertion to ensure the accuracy is in a plausible range.
    assert 0.0 <= acc <= 1.0, "Computed accuracy should be between 0 and 1."
    print("test_full_integration_on_real_dataset passed.")




if __name__ == "__main__":
    print("________________________________________________________ Test 1 _______________________")
    test_tokenize_training_example()
    print("________________________________________________________ Test 2 _______________________")
    test_create_subsample_groups()
    print("________________________________________________________ Test 3 _______________________")
    test_compute_accuracy()
    print("________________________________________________________ Test 4 _______________________")
    # test_fine_tune_model_zero_shot()
    print("________________________________________________________ Test 5 _______________________")
    # test_fine_tune_model_with_training_data()
    test_custom_data_collator()
    print("________________________________________________________ Test 6 _______________________")
    test_full_integration_on_real_dataset()
    print("All tests passed.")