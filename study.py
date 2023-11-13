# Plot the confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from functions import *

def plot_confusion_matrix(y_preds, y_true, labels, fn = "cm.png"):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    col_num = cm.shape[1]
    print(col_num)
    fig, ax = plt.subplots(figsize=(col_num+3, col_num+3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.savefig(fn)
    plt.show()


df_train = pd.read_csv("data/train-fr-sampled.txt", sep=",")
df_val = pd.read_csv("data/validation-fr-sampled.txt", sep=",")
df_test = pd.read_csv("data/test-fr-sampled.txt", sep=",")


ds = create_ds(df_train=df_train, df_val=df_val,df_test=df_test,text_col="review", label_col="label")
print(ds)

def tokenize(batch):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(batch["text"], padding=True, truncation=True)

model_name="output-nlpbaselines-10/checkpoint-320"
# model_name="camembert-base"

ds_encoded = ds.dataset.map(tokenize, batched=True, batch_size=None)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("output-nlpbaselines-10/checkpoint-320", num_labels=2).to(device)

trainer2 = Trainer(model,compute_metrics=compute_metrics)
preds_output_val = trainer2.predict(ds_encoded["validation"])
y_preds = np.argmax(preds_output_val.predictions, axis=1)
y_valid = np.array(ds_encoded["validation"]["label"])
labels = ["negatif","positif"]

plot_confusion_matrix(y_preds, y_valid, labels)