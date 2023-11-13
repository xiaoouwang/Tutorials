# load a model, run tests and report metrics
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import trainer_seq2seq
from functions import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("output-nlpbaselines-10/checkpoint-320", num_labels=2).to(device)
model_name="camembert-base"

def tokenize(batch):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(batch["text"], padding=True, truncation=True)

df_test = pd.read_csv("data/test-fr-sampled.txt", sep=",")
ds = create_ds(df_train=df_test,text_col="review", label_col="label")
print(ds)

ds_encoded = ds.dataset.map(tokenize, batched=True, batch_size=None)

trainer2 = Trainer(model,compute_metrics=compute_metrics)
preds_output_test = trainer2.predict(ds_encoded["test"])

for k,v in preds_output_test.metrics.items():
    print(k,v)