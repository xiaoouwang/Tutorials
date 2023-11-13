Design a class called classifier, it takes a bert-like model using AutoTokenizer and AutoModelForSequenceClassification and a dataset in csv as input, fine-tune the model on gpu and records the f1, accuracy, recall score on a property called f1, accuracy and recall

Implement the cross validation into it

Add a statistical test to test the significance of the f1 score

Use this code to compare bert VS. Distill bert on a toy dataset

Modify the code by imagine the dataset is in the format of csv

Implement paired t test and Wilcoxon into the class according to the result of Shapiro test on the distribution of the f1 scores, recall and accuracy scores. Record the result of Shapiro test and the significance level for the two tests. Don't choose the test, just record both p values while recording also the test that is recommended

Can you implement hyperparameter search into the training part?

Integrate early stopping, leave the choice to the user

Adapt the label number according to the number of the labels in the dataset

Give me the whole version of the script

How to implement a 2 GPU user case?

