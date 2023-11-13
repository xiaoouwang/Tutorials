from nlpbaselines import utils
import torch
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from argparse import ArgumentParser
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, recall_score
import pandas as pd
import numpy as np

def tokenize(batch, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == "camembert/camembert-base-ccnet-4gb":
        print("tokenize with ccnet")
        return tokenizer(batch["text"], padding=512, truncation=512)
    else:
        return tokenizer(batch["label"], padding=True, truncation=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average="weighted")
    return {"accuracy": acc, "recall":recall, "f1": f1}


def create_ds(df_train, df_val=None, df_test=None,text_col= "text",label_col="label"):
    from collections import namedtuple
    output = namedtuple('Dataset', ['dataset', 'n_labels'])
    ds = DatasetDict()
    # only the test that I said is available
    if df_val is None and df_test is None:
        test = Dataset.from_pandas(df_train)
        ds["test"] = test
    elif df_test is None:
        train,val = Dataset.from_pandas(df_train),Dataset.from_pandas(df_val)
        ds["train"],ds["validation"] = train, val
    else:
        train,val,test = Dataset.from_pandas(df_train),Dataset.from_pandas(df_val),Dataset.from_pandas(df_test)
        ds["train"],ds["validation"],ds["test"] = train, val, test
    n_label = df_train[label_col].nunique()
    ds = ds.class_encode_column(label_col)
    def modify_features(example):
        example["text"] = example[text_col]
        del example[text_col]
        return example
    return output(ds.map(modify_features), n_label)

def train(ds,model_name="camembert-base",batch_size=10,num_labels=2,epoch = 2, seed=40 , learning_rate = 2e-5, output="output-nlpbaselines"):
    def tokenize_train(batch):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if "camembert" in model_name:
            return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)
            print("tokenize with ccnet")
        else:
            return tokenizer(batch["text"], padding=True, truncation=True)
    ds_encoded = ds.dataset.map(tokenize_train, batched=True, batch_size=None)
    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=ds.n_labels
    ).to(device)
    logging_steps = len(ds_encoded["train"]) // batch_size
    output = f"{output}-{batch_size}"
    training_args = TrainingArguments(
        output_dir=output,
        num_train_epochs=epoch,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        # torch_compile=True,  # optimizations
        # optim="adamw_torch_fused", # improved optimizer
        push_to_hub=False,
        seed=seed,
        log_level="error",
    )

    training_args.set_save(strategy="epoch")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=ds_encoded["train"],
        eval_dataset=ds_encoded["validation"],
        tokenizer=tokenizer
    )

    trainer.train()
    return trainer