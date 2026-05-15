from torch.utils.data import Dataset


class TrafficSlidingWindowDataset(Dataset):
    """
    Tüm x_seq verisini baştan oluşturmaz.
    DataLoader örnek istedikçe geçmiş zaman penceresini anlık üretir.

    data_x shape:
        [Zaman, Düğüm, Özellik]
    """

    def __init__(self, data_x, window_size, target_col_indices, start_index, end_index):
        self.data_x = data_x
        self.window_size = window_size
        self.target_col_indices = target_col_indices
        self.start_index = start_index
        self.end_index = end_index

    def __len__(self):
        return self.end_index - self.start_index - self.window_size

    def __getitem__(self, index):
        real_index = self.start_index + index

        # X: geçmiş window_size kadar zaman adımı
        x_window = self.data_x[
            real_index : real_index + self.window_size
        ]

        # y: pencerenin hemen sonrasındaki zaman adımı
        y_target = self.data_x[
            real_index + self.window_size,
            :,
            self.target_col_indices
        ]

        return x_window, y_target