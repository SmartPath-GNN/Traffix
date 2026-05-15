import torch


def get_device():
    """
    CUDA varsa NVIDIA GPU kullanılır.
    Apple Silicon cihazlarda MPS varsa MPS kullanılır.
    Hiçbiri yoksa CPU kullanılır.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class EarlyStopping:
    """
    Validation loss iyileşmeyi bıraktığında eğitimi durdurur.
    En iyi modeli best_gnn_lstm_model.pt olarak kaydeder.
    """

    def __init__(self, patience=10, delta=0.0001, path="best_gnn_lstm_model.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"Early Stopping Sayacı: {self.counter} / {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        print("Yeni en iyi model kaydedildi.")