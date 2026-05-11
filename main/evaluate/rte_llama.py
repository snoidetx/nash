from dataclasses import dataclass, field, fields, MISSING
from main.utils.roots import setup_roots

setup_roots()

from copy import deepcopy
from typing import List

import numpy as np
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments

from ntk import *
from probe import *
from dvutils.Data_Shapley import *
from main.shapley.helpers.helper_freeshap import *
from main.utils.random import set_random_seed
from main.models.rte import RTELlama


DATASET_CONFIG_PATH = "main/configs/rte-llama.yaml"
N_TRAIN = 2490
N_VAL = 277
N_CLASSES = 2


def evaluate_rte_subset(indices: List[int]=np.arange(N_TRAIN),
                       device=None,
                       seed=None):
    if not device:
        device = torch.device('cpu')
    if seed:
        set_random_seed(seed)

    yaml_args = yaml.load(open(DATASET_CONFIG_PATH), Loader=yaml.Loader)
    list_dataset = yaml_args['dataset']
    train_subset = list_dataset.get_idx_dataset(indices, split="train")
    val_full = list_dataset.get_idx_dataset(np.arange(N_VAL), split="val")

    model = RTELlama(wrapped=True)
    #model.probe.args['device'] = device
    model.probe.model = model.probe.model.to(device)

    args = TrainingArguments(
        output_dir="saved_data/rte_llama",
        seed=seed,
        num_train_epochs=5,                  # bump to 3–5 if you want
        per_device_train_batch_size=2,
        per_device_eval_batch_size=32,
        learning_rate=0.00001,
        #weight_decay=0.01,
        #save_strategy="no",
        logging_steps=50,
        report_to=[],
        remove_unused_columns = False,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_strategy="epoch",
        save_total_limit=2 
    )

    def data_collator(batch):
        keys = batch[0].keys()
        out = {}
        for k in keys:
            vals = [b[k] for b in batch]
            out["labels" if k=="label" else k] = torch.stack(vals) if torch.is_tensor(vals[0]) else vals
        return out

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = (preds == labels).mean().item()
        return {"accuracy": 100*acc}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_subset,
        eval_dataset=val_full,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    metrics = trainer.evaluate()

    return metrics
