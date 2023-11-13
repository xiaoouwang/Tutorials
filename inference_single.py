# Run inference on a single item
#hide_output
from transformers import pipeline
import matplotlib.pyplot as plt
import pandas as pd

labels = ["negatif","positif"]

# Change `transformersbook` to your Hub username
model_id = "output-nlpbaselines-10/checkpoint-320"
classifier = pipeline("text-classification", model=model_id)
custom_tweet = "C'est pas trop mal."
preds = classifier(custom_tweet, return_all_scores=True)

preds_df = pd.DataFrame(preds[0])
plt.bar(labels, 100 * preds_df["score"], color='C0')
plt.title(f'"{custom_tweet}"')
plt.ylabel("Class probability (%)")
plt.savefig('fig1.png')
plt.show()
# save the figure

custom_tweet = "C'est nul."
preds = classifier(custom_tweet, return_all_scores=True)

preds_df = pd.DataFrame(preds[0])
plt.bar(labels, 100 * preds_df["score"], color='C0')
plt.title(f'"{custom_tweet}"')
plt.ylabel("Class probability (%)")
plt.savefig('fig2.png')
plt.show()