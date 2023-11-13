# Look for biggest loss
from torch.nn.functional import cross_entropy
import torch
import pandas as pd
from functions import *


def forward_pass_with_label(batch):
    # Place all input tensors on the same device as the model
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}

    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device),
                             reduction="none")

    # Place outputs on CPU for compatibility with other dataset columns
    return {"loss": loss.cpu().numpy(),
            "predicted_label": pred_label.cpu().numpy()}

def label_int2str(row):
    return {0:'negatif', 1: 'positif'}[row]

#hide_output

def tokenize(batch):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(batch["text"], padding=True, truncation=True)

model_name="camembert-base"

df_train = pd.read_csv("data/train-fr-sampled.txt", sep=",")
df_val = pd.read_csv("data/validation-fr-sampled.txt", sep=",")
df_test = pd.read_csv("data/test-fr-sampled.txt", sep=",")


ds = create_ds(df_train=df_train, df_val=df_val,df_test=df_test,text_col="review", label_col="label")
print(ds)

ds_encoded = ds.dataset.map(tokenize, batched=True, batch_size=None)
model_name="output-nlpbaselines-10/checkpoint-320"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# define tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Convert our dataset back to PyTorch tensors
ds_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])
# Compute loss values
ds_encoded["validation"] = ds_encoded["validation"].map(
    forward_pass_with_label, batched=True, batch_size=16)

ds_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = ds_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (df_test["predicted_label"]
                              .apply(label_int2str))

# highest losses
# print(df_test.sort_values("loss", ascending=False).head(10))
df_test.sort_values("loss", ascending=False).to_csv("data/highest_losses.csv", index=False)

# lowest losses
df_test.sort_values("loss", ascending=True).to_csv("data/lowest_losses.csv", index=False)