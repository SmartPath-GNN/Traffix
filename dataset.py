from torch.utils.data import Dataset
import torch

class TrafficSlidingWindowDataset(Dataset):
    """
    Kısa dönem (recent), günlük (daily) ve haftalık (weekly) zaman pencerelerini
    ve horizon boyunca gelecek zaman dizisini anlık üretir.
    """
    def __init__(
        self,
        data_x,
        window_size,  # Kısa dönem / recent pencere boyutu (örn: 3)
        target_col_indices,
        start_index,
        end_index,
        horizon=6     # Tahmin ufku boyunca üretilecek dizi adımı
    ):
        self.data_x = data_x
        self.window_size = window_size
        self.target_col_indices = target_col_indices
        self.start_index = start_index  # Güvenli indeks artık train.py'den geliyor
        self.end_index = end_index
        self.horizon = horizon

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, index):
        t = self.start_index + index

        # 1. Kısa Dönem (Recent)
        x_recent = self.data_x[t - self.window_size : t]
        
        # 2. Günlük Dönem (Daily)
        x_daily = self.data_x[t - 24 - (self.window_size // 2) : t - 24 + (self.window_size // 2) + (self.window_size % 2)]
        
        # 3. Haftalık Dönem (Weekly)
        x_weekly = self.data_x[t - 168 - (self.window_size // 2) : t - 168 + (self.window_size // 2) + (self.window_size % 2)]

        # Y: Horizon boyunca hedef zaman dizisi
        y_target = self.data_x[t : t + self.horizon, :, self.target_col_indices]

        return x_recent, x_daily, x_weekly, y_target