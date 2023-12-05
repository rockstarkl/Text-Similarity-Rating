# Import necessary libraries
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd


# Define the Binary Classification dataset class
class BinaryClassificationDataset(Dataset):
    def __init__(self, sentences1, sentences2, labels, tokenizer, max_len):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences1)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences1[idx],
            self.sentences2[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# Define the Binary Classification model
# Define the Binary Classification model
class BinaryClassificationModel(nn.Module):
    def __init__(self):
        super(BinaryClassificationModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_states, dim=1)  # Use mean pooling
        logits = self.fc(pooled_output)
        return logits


# Function to train the model
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits.flatten(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# Function to evaluate the model
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits.flatten(), labels)

            total_loss += loss.item()

    return total_loss / len(val_loader)



csv_file_path = 'C:/Users/kuldeep.limbachiya/Downloads/DataNeuron_Task/Precily_Task/Precily_Text_Similarity_1.csv'
df = pd.read_csv(csv_file_path)

# Assuming your Excel sheet has columns named 'text1', 'text2', and 'label'
sentences1 = df['text1'].astype(str).tolist()
sentences2 = df['text2'].astype(str).tolist()
labels = df['SimilarityRating'].tolist()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_len = 128
batch_size = 4

dataset = BinaryClassificationDataset(sentences1, sentences2, labels, tokenizer, max_len)

train_size = int(0.5 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = BinaryClassificationModel()
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits

device = torch.device("cpu")
model.to(device)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):

    train_loss = train_model(model, train_loader, optimizer, criterion, device)

    train_loss = train_model(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate_model(model, val_loader, criterion, device)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'binary_classification_model.pth')
