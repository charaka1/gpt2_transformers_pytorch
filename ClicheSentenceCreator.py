"""
This code fine-tunes a pretrained GPT-2 model on the provided sentences and then generates new sentences using the fine-tuned model. 
Code uses the Hugging Face Transformer library to access its GPT2 model and the tokenizer
"""
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

# Read sentences from external sentences data file
with open('cliche_sentences.txt', 'r') as file:
    sentences = file.readlines()

# Use the sequence tokenizer from the Hugging Face Transformer library
# Tokenizer converts words in sentences to their numerical format that GPT2 expects
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Use the GPT2 (Transformer) model from the Hugging Face Transformer library
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Tokenize the sentences to the format that GPT2 expects
tokenized_text = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in sentences]

# Custom SentenceDataset class extends the abstract PyTorch Dataset class to overide 3 methods:
# initializer to set the tokenized text, len method to return number of sentences and 
# getitem to retrieve a tokenized sentence as a Pytorch tensor.
class SentenceDataset(Dataset):
    def __init__(self, tokenized_text):
        self.tokenized_text = tokenized_text

    def __len__(self):
        return len(self.tokenized_text)

    def __getitem__(self, idx):
        return torch.tensor(self.tokenized_text[idx])

# Create PyTorch dataset and dataloader objects
dataset = SentenceDataset(tokenized_text)
# Since the batch_size is set to 1, dataloader will return one tokenized setence at a time to the model
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Prepare Transformer model for training
# Use a GPU for computation if one is available or else use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the model to GPU 
model.to(device)
# Set the model in training mode:
# Note, some layers in a model, like dropout and batch normalization, behave differently during training and evaluation
model.train()

# Setup configuration for updating model's parameters during training:
#AdamW implements a variant of Adam optimizer that is effective for Transformer models
optimizer = AdamW(model.parameters(), lr=5e-5)
# Learning rate scheduler linearly decrease the learning rate set in the optimizer to zero
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader)*5)

# Number of training epochs
num_epochs = 10

# Main training loop for thr GPT2 model
for epoch in range(num_epochs): 
    total_loss = 0
    # Iterate over batches of data in the dataloader
    for batch in dataloader:
        # In this example inputs and labels are the same (typical for unsupervised learning task like language modeling)
        inputs, labels = (batch, batch)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Model returns outputs object to extract loss
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Set gradients to zero and perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        # update the model's parameters based on the computed gradients in backpropagation
        optimizer.step()
        # Update the learning rate
        scheduler.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Set the model in evaluation mode
model.eval()

# Generate new sentences
i = 1
for _ in range(2): 
    input_ids = tokenizer.encode("Life is", return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=1.0, pad_token_id=tokenizer.eos_token_id)
    generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Sentence" + str(i) + ":" + generated_sentence)
    i += 1
