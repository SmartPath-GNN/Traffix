import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

def repeat_edge_index_for_batch(edge_index, edge_weight, batch_size, node_count, device):
    """Her batch örneği için düğüm ID'lerine offset ekleyerek grafı çoğaltır."""
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

class MultiStreamGNNLSTM(nn.Module):
    """
    Gelişmiş Çok Kollu Spatio-Temporal Model:
    Recent, Daily ve Weekly kolları mekânsal ilişkileri GNN katmanıyla ortak öğrenir.
    Oversmoothing ve Varyans Çökmesine karşı LayerNorm ve Sigmoid Gate içerir.
    """
    def __init__(self, input_features, gnn_hidden=16, lstm_hidden=32, output_features=2, horizon=6, dropout=0.2):
        super(MultiStreamGNNLSTM, self).__init__()
        self.horizon = horizon

        # 1. BOYUT EŞİTLEYİCİ PROJEKSİYON KATMANI
        # Kavşağın kendi orijinal kimliğini (hızını) koruyabilmesi için GNN boyutuna çıkarıyoruz.
        self.input_proj = nn.Linear(input_features, gnn_hidden)

        # Ortak Graf Evrişim Katmanı
        self.gnn = GCNConv(gnn_hidden, gnn_hidden)
        
        # 🌟 YENİ 1: Sinyal sönümlenmesini ve ortalamaya çökmeyi engelleyen LayerNorm'lar
        self.norm_gnn = nn.LayerNorm(gnn_hidden)
        self.norm_fused = nn.LayerNorm(lstm_hidden * 3)

        # Alpha başlangıçta 0'a yakın (Kavşak kendi öz verisine daha çok güvensin)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.gnn_dropout = nn.Dropout(p=dropout)

        # Bağımsız Zamansal Bellek Kolları
        self.lstm_recent = nn.LSTM(input_size=gnn_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.lstm_daily  = nn.LSTM(input_size=gnn_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.lstm_weekly = nn.LSTM(input_size=gnn_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.lstm_dropout = nn.Dropout(p=dropout)

        # Birleştirilmiş doğrusal çıktı katmanı (3 kol * lstm_hidden)
        self.linear = nn.Linear(lstm_hidden * 3, output_features * self.horizon)

    def _process_stream(self, x_seq, edge_index, edge_weight, lstm_layer, batch_size, node_count, in_features):
        time_steps = x_seq.shape[1]
        batch_edge_index, batch_edge_weight = repeat_edge_index_for_batch(
            edge_index, edge_weight, batch_size, node_count, x_seq.device
        )

        gnn_outputs = []
        
        # 🌟 YENİ 2: Alpha değerini kesin olarak 0 ile 1 arasına sıkıştıran Sigmoid kapısı
        gate = torch.sigmoid(self.alpha)

        for t in range(time_steps):
            x_t_flat = x_seq[:, t, :, :].reshape(batch_size * node_count, in_features)
            
            # Girdiyi GNN boyutuna yansıt (Kavşağın orijinal kimliği)
            proj_x = self.input_proj(x_t_flat)
            
            # Komşularla harmanla (GNN'in ürettiği bulamaç)
            h_t = self.gnn(proj_x, batch_edge_index, edge_weight=batch_edge_weight)
            
            # 3. OVERSMOOTHING KALKANI (Sigmoid Kapılı Residual Connection)
            # Gate ile harmanlanmış GNN çıktısı ve (1-Gate) ile kavşağın kendi ham verisini topluyoruz.
            h_t = gate * h_t + (1 - gate) * proj_x
            
            # 🌟 YENİ 3: LayerNorm Uygulaması (Her sokağın matematiksel varyansını canlı tutar)
            h_t = self.norm_gnn(h_t)
            h_t = torch.relu(h_t)
            
            h_t = self.gnn_dropout(h_t)
            gnn_outputs.append(h_t.reshape(batch_size, node_count, -1))

        gnn_out = torch.stack(gnn_outputs, dim=1).permute(0, 2, 1, 3)
        gnn_out = gnn_out.reshape(batch_size * node_count, time_steps, -1)
        
        lstm_out, _ = lstm_layer(gnn_out)
        return lstm_out[:, -1, :]

    def forward(self, x_recent, x_daily, x_weekly, edge_index, edge_weight):
        batch_size, _, node_count, in_features = x_recent.shape

        h_recent = self._process_stream(x_recent, edge_index, edge_weight, self.lstm_recent, batch_size, node_count, in_features)
        h_daily  = self._process_stream(x_daily, edge_index, edge_weight, self.lstm_daily, batch_size, node_count, in_features)
        h_weekly = self._process_stream(x_weekly, edge_index, edge_weight, self.lstm_weekly, batch_size, node_count, in_features)

        # Kolları yan yana bağlama (Late Fusion)
        fused_features = torch.cat([h_recent, h_daily, h_weekly], dim=-1)
        
        # 🌟 YENİ 4: LSTM çıkışında da LayerNorm (Lineer katmanın aynı sayıları basmasını engeller)
        fused_features = self.norm_fused(fused_features)
        
        fused_features = self.lstm_dropout(fused_features)

        # Tahmin üretimi ve 4D formata dönüştürme -> [Batch, Horizon, Düğüm, Özellik]
        prediction = self.linear(fused_features)
        prediction = prediction.reshape(batch_size, node_count, self.horizon, -1).permute(0, 2, 1, 3)
        
        return prediction