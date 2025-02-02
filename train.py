# train.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from evaluate import compute_accuracy
from data_collator import custom_data_collator

def fine_tune_model(model_name, train_dataset, eval_dataset, output_dir, training_args):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # Can be None for zero-shot.
        eval_dataset=eval_dataset,      # Passed for evaluation.
        tokenizer=tokenizer,
        data_collator=lambda features: custom_data_collator(features, tokenizer),
    )
    
    if train_dataset is not None:
        trainer.train()
    
    accuracy = compute_accuracy(model, eval_dataset, tokenizer)
    return accuracy
