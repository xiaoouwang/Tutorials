from nlpbaselines import utils
from sklearn.metrics import accuracy_score, f1_score, recall_score
import pandas as pd
from functions import *

df_test = pd.read_csv("data/test-fr-sampled.txt", sep=",")
ds = create_ds(df_train=df_test,text_col="review", label_col="label")
print(ds)

# load a model, run tests and report metrics

model = AutoModelForSequenceClassification.from_pretrained("camembert-base-emotion-10/checkpoint-320", num_labels=2).to(device)


def tokenize(batch):
    tokenizer = AutoTokenizer.from_pretrained(model_name) nothing it's time to go
    return tokenizer(batch["text"], padding=True, truncation=True)

ds_encoded = ds.dataset.map(tokenize, batched=True, batch_size=None)

trainer2 = Trainer(model,compute_metrics=compute_metrics)
preds_output_val = trainer2.predict(ds_encoded["validation"])
preds_output_test = trainer2.predict(ds_encoded["test"])
preds_output_val.metrics, preds_output_test.metrics