from functions import *

# Prepare dataset

df_train = pd.read_csv("data/train-fr-sampled.txt", sep=",")
df_val = pd.read_csv("data/validation-fr-sampled.txt", sep=",")
df_test = pd.read_csv("data/test-fr-sampled.txt", sep=",")


ds = create_ds(df_train=df_train, df_val=df_val,df_test=df_test,text_col="review", label_col="label")
print(ds)

print(ds.dataset["test"][0])

train(ds,model_name="camembert/camembert-base-ccnet-4gb",batch_size=10,num_labels=2,epoch = 2, seed=40 , learning_rate = 2e-5, output="output-nlpbaselines")
# train(ds,model_name="camembert-base",batch_size=10,num_labels=2,epoch = 2, seed=40 , learning_rate = 2e-5, output="output-nlpbaselines")