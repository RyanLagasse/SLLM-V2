# train.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from evaluate import compute_accuracy, compute_brimi_metrics
from data_collator import custom_data_collator

def fine_tune_model(model_name, train_dataset, eval_dataset, output_dir, training_args, eval_fn="mmlu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=lambda features: custom_data_collator(features, tokenizer),
    )
    
    if train_dataset is not None:
        trainer.train()
    
    # Use the appropriate evaluation function.
    if eval_fn == "brimi":
        metrics = compute_brimi_metrics(model, eval_dataset, tokenizer)
    else:
        # For MMLU, we can use compute_accuracy (or a similar function).
        acc = compute_accuracy(model, eval_dataset, tokenizer)
        metrics = {"accuracy": acc}
    return metrics
