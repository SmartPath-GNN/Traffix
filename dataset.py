from torch.utils.data import Dataset


class TrafficSlidingWindowDataset(Dataset):
    """
    Tüm x_seq verisini baştan oluşturmaz.
    DataLoader örnek istedikçe geçmiş zaman penceresini anlık üretir.

    data_x shape:
        [Zaman, Düğüm, Özellik]
    """

    def __init__(
        self,
        data_x,
        window_size,
        target_col_indices,
        start_index,
        end_index,
        horizon=1
    ):
        self.data_x = data_x
        self.window_size = window_size
        self.target_col_indices = target_col_indices
        self.start_index = start_index
        self.end_index = end_index
        self.horizon = horizon

    def __len__(self):
        return self.end_index - self.start_index - self.window_size - self.horizon + 1

    def __getitem__(self, index):
        real_index = self.start_index + index

        # X: geçmiş window_size kadar zaman adımı
        x_window = self.data_x[
            real_index : real_index + self.window_size
        ]

        # y: pencerenin horizon kadar sonrasındaki zaman adımı
        # horizon=1 ise hemen sonraki adımı tahmin eder.
        # horizon=3 ise 3 adım sonrasını tahmin eder.
        target_index = real_index + self.window_size + self.horizon - 1

        y_target = self.data_x[
            target_index,
            :,
            self.target_col_indices
        ]

        return x_window, y_target