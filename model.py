import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


def repeat_edge_index_for_batch(edge_index, edge_weight, batch_size, node_count, device):
    """
    Batch içinde aynı yol grafını birden fazla örnek için çoğaltır.
    Her batch örneği için düğüm ID'lerine offset eklenir.
    """

    edge_indices = []
    edge_weights = []

    for batch_id in range(batch_size):
        offset = batch_id * node_count

        shifted_edge_index = edge_index + offset

        edge_indices.append(shifted_edge_index)
        edge_weights.append(edge_weight)

    batch_edge_index = torch.cat(edge_indices, dim=1).to(device)
    batch_edge_weight = torch.cat(edge_weights, dim=0).to(device)

    return batch_edge_index, batch_edge_weight


class GNNLSTM(nn.Module):
    """
    GCNConv yol ağı üzerindeki mekânsal ilişkileri öğrenir.
    LSTM zaman içerisindeki trafik değişimini öğrenir.
    Linear katman bir sonraki zaman adımı için tahmin üretir.
    """

    def __init__(self, input_features, gnn_hidden, lstm_hidden, output_features,dropout=0.1):
        super(GNNLSTM, self).__init__()

        self.gnn = GCNConv(input_features, gnn_hidden)

         # GNN çıktısından sonra dropout
        self.gnn_dropout = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(
            input_size=gnn_hidden,
            hidden_size=lstm_hidden,
            batch_first=True
        )
         # LSTM çıktısından sonra dropout
        self.lstm_dropout = nn.Dropout(p=dropout)

        self.linear = nn.Linear(lstm_hidden, output_features)

    def forward(self, x_seq, edge_index, edge_weight):
        batch_size, time_steps, node_count, in_features = x_seq.shape

        batch_edge_index, batch_edge_weight = repeat_edge_index_for_batch(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch_size=batch_size,
            node_count=node_count,
            device=x_seq.device
        )

        gnn_outputs = []

        for t in range(time_steps):
            x_t = x_seq[:, t, :, :]

            x_t_flat = x_t.reshape(batch_size * node_count, in_features)

            h_t = self.gnn(
                x_t_flat,
                batch_edge_index,
                edge_weight=batch_edge_weight
            )

            h_t = torch.relu(h_t)

             # GNN dropout
            h_t = self.gnn_dropout(h_t)

            h_t = h_t.reshape(batch_size, node_count, -1)

            gnn_outputs.append(h_t)

        gnn_out = torch.stack(gnn_outputs, dim=1)

        gnn_out = gnn_out.permute(0, 2, 1, 3)
        gnn_out = gnn_out.reshape(batch_size * node_count, time_steps, -1)

        lstm_out, _ = self.lstm(gnn_out)

        last_hidden = lstm_out[:, -1, :]

        # LSTM dropout
        last_hidden = self.lstm_dropout(last_hidden)

        prediction = self.linear(last_hidden)

        prediction = prediction.reshape(batch_size, node_count, -1)

        return prediction