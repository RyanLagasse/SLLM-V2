# data_collator.py
from typing import List, Dict, Any
import torch
from transformers import PreTrainedTokenizerBase

def custom_data_collator(features: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase):
    """
    Custom data collator that explicitly pads the following keys:
      - "input_ids" (using tokenizer.pad_token_id)
      - "attention_mask" (padding with 0)
      - "labels" (padding with -100, the ignore index)
    It then stacks any remaining keys.
    """
    # print("Features received by custom_data_collator:", features)

    pad_keys = ["input_ids", "attention_mask", "labels"]
    # Determine maximum length for the pad_keys.
    max_length = max(len(f["input_ids"]) for f in features)
    
    # Pad the pad_keys explicitly.
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for f in features:
        cur_len = len(f["input_ids"])
        pad_len = max_length - cur_len
        
        # Pad input_ids
        padded_input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        # Pad attention_mask
        padded_attention_mask.append(f["attention_mask"] + [0] * pad_len)
        # Pad labels with -100
        padded_labels.append(f["labels"] + [-100] * pad_len)
    
    batch = {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long)
    }
    
    # For any remaining keys that are present in the first feature but not in pad_keys:
    for key in features[0]:
        if key not in pad_keys:
            try:
                batch[key] = torch.tensor([f[key] for f in features])
            except Exception:
                batch[key] = [f[key] for f in features]
    
    return batch
