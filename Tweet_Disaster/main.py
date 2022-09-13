import torch
import torch.nn as nn
import torch.optim as optim
from models.LSTM_model import LSTMTweet
from utils.preprocessing import preprocessing_text, load_file
from tqdm import tqdm as tqdm_notebook


path = 'data/'
batch_size = 32
max_document_len = 100
embeding_size = 300
hidden_size = 100
max_size = 5000
device = 'cpu'

## load dataset
tokenize = lambda x: x.split()
vocab_size, train_iterator, valid_iterator, test_iterator, Text, Label = load_file(path, tokenize,\
                                                                       batch_size, max_document_len, max_size, device)
# model and optimizer defination.                                                                
model = LSTMTweet(vocab_size, embeding_size, hidden_size,device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.05, momentum=0.9)

def accuracy(prediction, actual):
    return 100* torch.sum([prediction.argmax(dim = 1)== actual])

def train(model, iterator, optimizer, criterion):
    total_loss = 0
    total_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grab()
        text = batch.text[0]
        preds = model(text)
        acc = accuracy(preds, batch.target)
        loss = criterion(preds, batch.target.squeeze())

        # back propagation
        loss.backward()
        optimizer.step()

        total_loss += loss
        total_acc +=acc
    return total_loss/len(iterator), total_acc/len(iterator)
def evaluate(model, iterator, criterion):
    total_loss = 0
    total_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text = batch.text[0]
            preds = model(text)
            acc = accuracy(preds, batch.target)
            loss = criterion(preds, batch.target.squeeze())
            total_loss += loss
            total_acc +=acc
    return total_loss/len(iterator), total_acc/len(iterator)

def train_model(epochs, model, train_iterator, valid_iterator, optimizer, criterion):
    best_acc = 0
    for epoch in epochs:
        # train model
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        # evaluate model
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        # save best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), 'saved_weights.pt')
        print('Epoch {}/{}: train_loss = {}, train_acc = {}\t\t valid_loss = {}, valid_acc = {}'.format(epoch, epochs, train_loss, train_acc, valid_acc, valid_acc))
    return best_acc


best_acc = train_model(50 ,model, train_iterator, valid_iterator, optimizer, criterion)
print(best_acc)
## predict test.
# load model
state = torch.load('saved_weights.pt')
model.load_state_dict(state['state_dict'],strict=True)
loss, acc = evaluate(model, test_iterator, criterion)
print('Test acc: ', acc)