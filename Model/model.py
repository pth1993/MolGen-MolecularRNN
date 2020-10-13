import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *


class NodeRNN(nn.Module):
    def __init__(self, input_size_adj, input_size_node, emb_size, hidden_lstm_size, num_layers, hidden_header_size,
                 out_features, seq_len, dr_rate=0.2):
        super(NodeRNN, self).__init__()
        self.seq_len = seq_len
        self.embedding_adj = nn.Embedding(input_size_adj, emb_size, padding_idx=0)
        self.embedding_node = nn.Embedding(input_size_node, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_lstm_size, num_layers=num_layers,
                            dropout=dr_rate, batch_first=True)
        self.header = nn.Sequential(
            nn.Linear(in_features=hidden_lstm_size, out_features=hidden_header_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_header_size, out_features=out_features)
        )

    def forward(self, x_adj, x_node, x_len):
        h_S = self.embedding(x_adj)
        h_C = self.embedding(x_node)
        h = torch.cat((h_S, h_C), dim=1)
        h = torch.nn.utils.rnn.pack_padded_sequence(h, x_len, batch_first=True, enforce_sorted=False)
        h, state = self.lstm_fr(h)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=self.fr_len)
        outputs = []
        for i in range(self.seq_len):
            y = self.header(h[:, i])
            outputs.append(y)
        outputs = torch.stack(outputs, dim=1)

        return outputs, state


class EdgeRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_lstm_size, num_layers, hidden_header_size, out_features,
                 seq_len, dr_rate=0.2):
        super(EdgeRNN, self).__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(input_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_lstm_size, num_layers=num_layers,
                            dropout=dr_rate, batch_first=True)
        self.header = nn.Sequential(
            nn.Linear(in_features=hidden_lstm_size, out_features=hidden_header_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_header_size, out_features=out_features)
        )

    def forward(self, x, x_len, state):
        h = self.embedding(x)
        h = torch.nn.utils.rnn.pack_padded_sequence(h, x_len, batch_first=True, enforce_sorted=False)
        h, state = self.lstm_fr(h, state)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=self.fr_len)
        outputs = []
        for i in range(self.seq_len):
            y = self.header(h[:, i])
            outputs.append(y)
        outputs = torch.stack(outputs, dim=1)
        return outputs


if __name__ == "__main__":
    model = NodeRNN(input_size_adj=12, input_size_node=9, emb_size=64, hidden_lstm_size=128, num_layers=2,
                    hidden_header_size=64, out_features=9, seq_len=40)

    x1 = np.ones([16, 80])
    for i in range(x1.shape[0]):
        for j in range(i):
            x1[i, x1.shape[1]-j-1] = 0

    np.random.shuffle(x1)
    x1_len = [int(np.sum(x1[i, :])) for i in range(x1.shape[0])]
    x2_len = [18 for i in range(x1.shape[0])]

    x1 = torch.tensor(x1, dtype=torch.long).to("cuda")
    x2 = torch.tensor(np.ones([16, 25]), dtype=torch.long).to("cuda")

    print(x1.shape, x2.shape)
    output = model(x1, x2, x1_len, x2_len)
    print(output.shape)
