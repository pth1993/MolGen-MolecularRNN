import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *


class NodeRNN(nn.Module):
    def __init__(self, input_size_adj, input_size_node, emb_size, hidden_rnn_size, hidden_header_size,
                 out_features, seq_len, dr_rate=0.2):
        super(NodeRNN, self).__init__()
        self.seq_len = seq_len
        self.embedding_adj = nn.Linear(input_size_adj, emb_size, bias=False)
        self.embedding_node = nn.Embedding(input_size_node, emb_size, padding_idx=0)
        self.rnn1 = nn.GRU(input_size=emb_size, hidden_size=hidden_rnn_size, num_layers=1, batch_first=True)
        self.rnn2 = nn.GRU(input_size=hidden_rnn_size, hidden_size=hidden_rnn_size, num_layers=1, batch_first=True)
        self.rnn3 = nn.GRU(input_size=hidden_rnn_size, hidden_size=hidden_rnn_size, num_layers=1, batch_first=True)
        self.rnn4 = nn.GRU(input_size=hidden_rnn_size, hidden_size=hidden_rnn_size, num_layers=1, batch_first=True)
        self.header = nn.Sequential(
            nn.Linear(in_features=hidden_rnn_size, out_features=hidden_header_size),
            nn.ReLU(),
            nn.Dropout(dr_rate),
            nn.Linear(in_features=hidden_header_size, out_features=out_features)
        )

    def forward(self, x_adj, x_node, x_len):
        h_S = []
        for i in range(self.seq_len):
            y = self.embedding_adj(x_adj[:, i])
            h_S.append(y)
        h_S = torch.stack(h_S, dim=1)
        h_C = self.embedding_node(x_node)
        h = torch.cat((h_S, h_C), dim=1)
        h = torch.nn.utils.rnn.pack_padded_sequence(h, x_len, batch_first=True, enforce_sorted=False)
        h1, _ = self.rnn1(h)
        h2, _ = self.rnn2(h1)
        h3, _ = self.rnn3(h2)
        h4, _ = self.rnn4(h3)
        h1, _ = torch.nn.utils.rnn.pad_packed_sequence(h1, batch_first=True, total_length=self.seq_len)
        h2, _ = torch.nn.utils.rnn.pad_packed_sequence(h2, batch_first=True, total_length=self.seq_len)
        h3, _ = torch.nn.utils.rnn.pad_packed_sequence(h3, batch_first=True, total_length=self.seq_len)
        h4, _ = torch.nn.utils.rnn.pad_packed_sequence(h4, batch_first=True, total_length=self.seq_len)
        outputs = torch.zeros([h4.shape[0], self.seq_len, len(ATOM_IDX)], device=device)
        for i in range(self.seq_len):
            y = self.header(h4[:, i])
            outputs[:, i, ] = y

        return outputs, torch.stack([h1, h2, h3, h4], dim=1)


class EdgeRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_rnn_size, hidden_header_size, out_features,
                 seq_len, dr_rate=0.2):
        super(EdgeRNN, self).__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn1 = nn.GRU(input_size=emb_size, hidden_size=256, num_layers=1, batch_first=True)
        self.rnn2 = nn.GRU(input_size=256, hidden_size=hidden_rnn_size, num_layers=1, batch_first=True)
        self.rnn3 = nn.GRU(input_size=hidden_rnn_size, hidden_size=hidden_rnn_size, num_layers=1, batch_first=True)
        self.rnn4 = nn.GRU(input_size=hidden_rnn_size, hidden_size=hidden_rnn_size, num_layers=1, batch_first=True)
        self.header = nn.Sequential(
            nn.Linear(in_features=hidden_rnn_size, out_features=hidden_header_size),
            nn.Dropout(dr_rate),
            nn.ReLU(),
            nn.Linear(in_features=hidden_header_size, out_features=out_features)
        )

    def forward(self, x, state):
        h = self.embedding(x)
        print(h.shape, state.shape)
        h, _ = self.rnn1(h, state[:, 0].contiguous().unsqueeze(0))
        print(h.shape)
        h, _ = self.rnn2(h, state[:, 1].contiguous().unsqueeze(0))
        h, _ = self.rnn3(h, state[:, 2].contiguous().unsqueeze(0))
        h, _ = self.rnn4(h, state[:, 3].contiguous().unsqueeze(0))

        outputs = torch.zeros([h.shape[0], self.seq_len, len(BOND_IDX)], device=device)
        for i in range(self.seq_len):
            y = self.header(h[:, i].contiguous())
            outputs[:, i, ] = y

        return outputs


if __name__ == "__main__":
    model = NodeRNN(input_size_adj=12, input_size_node=9, emb_size=64, hidden_rnn_size=128,
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
