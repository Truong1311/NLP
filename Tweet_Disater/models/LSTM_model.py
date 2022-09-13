import torch
import torch.nn as nn

class LSTMTweet(nn.Module):
    def __init__(self,vocab_size, embed_size, hidden_size, device, num_layers = 1):
        self.vocab_size = vocab_size
        self.embedding_size = embed_size
        self.hidden_size = hidden_size
        self.device = device
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embed_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first = True,
                            dropout = 0,
                            bidirectional = False)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(self.hidden_size, 1)
    def forward(self, x):
        # khai báo trạng thái ban đầu cho lstm cell
        h0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        c0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        ## khởi tạo giá trị phân phối chuẩn cho h0 và c0
        nn.init.xavier_normal_(h0)
        nn.init.xavier_normal_(c0)

        embedding = self.embedding(x)
        out, (hn, cn) = self.lstm(embedding, (h0, c0))
        dropout = self.dropout(hn[-1]) # lấy output của hidden layer cuối cùng.
        out = torch.sigmoid(self.fc(dropout))
        return out