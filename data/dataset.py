import torch


class TensorBatchLoader:
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor | None = None,
        batch_size: int = 2048,
        shuffle: bool = True,
    ) -> None:
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]
        assert batch_size > 0
        self.x = x.float()
        self.y = y.float()
        self.w = w.float() if w is not None else torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device)
        assert self.w.shape == (x.shape[0], 1)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (self.x.shape[0] + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = self.x.shape[0]
        if not self.shuffle:
            for i in range(0, n, self.batch_size):
                yield self.x[i : i + self.batch_size], self.y[i : i + self.batch_size], self.w[i : i + self.batch_size]
            return

        idx = torch.randperm(n, device=self.x.device)
        for i in range(0, n, self.batch_size):
            b = idx[i : i + self.batch_size]
            yield self.x[b], self.y[b], self.w[b]


def create_dataloader(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor | None = None,
    batch_size: int = 2048,
    shuffle: bool = True,
    device: torch.device | str = "cpu",
) -> TensorBatchLoader:
    w_dev = w.to(device) if w is not None else None
    return TensorBatchLoader(x.to(device), y.to(device), w_dev, batch_size=batch_size, shuffle=shuffle)


def build_dataloaders(
    x: torch.Tensor,
    y: torch.Tensor,
    train_split: float,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[TensorBatchLoader, TensorBatchLoader]:
    assert 0.0 < train_split < 1.0
    n_total = x.shape[0]
    n_train = int(n_total * train_split)
    assert 0 < n_train < n_total

    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).to(x.device)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    train_loader = create_dataloader(x[train_idx], y[train_idx], batch_size=batch_size, shuffle=True, device=device)
    val_loader = create_dataloader(x[val_idx], y[val_idx], batch_size=eval_batch_size, shuffle=False, device=device)
    return train_loader, val_loader
