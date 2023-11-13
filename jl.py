import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn as nn
import torch.nn.functional as F

# Toy dataset
# Sentence pair classification: 0 for not similar, 1 for similar
sentence_pairs = [
    ("The cat sat on the mat", "A cat was sitting on the mat", 1),
    ("He loves to play football", "Soccer is his favorite sport", 0),
    ("The sky is blue", "Water is clear", 0),
    ("Reading is a good habit", "She enjoys reading books", 1)
]

# Sentence classification: 0 for negative, 1 for positive
sentences = [
    ("The movie was fantastic", 1),
    ("I hated the meal", 0),
    ("What a wonderful day", 1),
    ("It was a terrible experience", 0)
]

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_pair(pair):
    return tokenizer(pair[0], pair[1], padding='max_length', truncation=True, max_length=128, return_tensors='pt')

def tokenize_sentence(sentence):
    return tokenizer(sentence, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

# Prepare data
X_sp = [tokenize_pair(pair) for pair, _, _ in sentence_pairs]
y_sp = torch.tensor([label for _, _, label in sentence_pairs])

X_s = [tokenize_sentence(sentence) for sentence, _ in sentences]
y_s = torch.tensor([label for _, label in sentences])

# Convert to TensorDataset
dataset_sp = TensorDataset(torch.stack([x['input_ids'].squeeze(0) for x in X_sp]),
                           torch.stack([x['attention_mask'].squeeze(0) for x in X_sp]), y_sp)

dataset_s = TensorDataset(torch.stack([x['input_ids'].squeeze(0) for x in X_s]),
                          torch.stack([x['attention_mask'].squeeze(0) for x in X_s]), y_s)

# Dataloaders
dataloader_sp = DataLoader(dataset_sp, batch_size=2)
dataloader_s = DataLoader(dataset_s, batch_size=2)

# Load pre-trained BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Define custom model
class JointBertModel(nn.Module):
    def __init__(self, bert_model):
        super(JointBertModel, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        return self.classifier(pooled_output)

# Initialize model
model = JointBertModel(bert_model)

# Optimizer
# optimizer = AdamW(model.parameters(), lr=1e-5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training
for epoch in range(2):  # loop over the dataset multiple times
    for batch_sp, batch_s in zip(dataloader_sp, dataloader_s):
        input_ids_sp, attention_mask_sp, labels_sp = batch_sp
        input_ids_s, attention_mask_s, labels_s = batch_s

        # Forward pass for sentence pair classification
        outputs_sp = model(input_ids_sp, attention_mask_sp)
        loss_sp = F.cross_entropy(outputs_sp, labels_sp)

        # Forward pass for sentence classification
        outputs_s = model(input_ids_s, attention_mask_s)
        loss_s = F.cross_entropy(outputs_s, labels_s)

        # Combine losses and update model parameters
        loss = loss_sp + loss_s
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# The model is now trained on the toy dataset for both tasks
"Model trained on toy dataset for joint learning tasks"
