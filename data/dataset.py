import torch
from torch.utils.data import DataLoader, Dataset, random_split


class OptionDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None) -> None:
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]
        self.x = x.float()
        self.y = y.float()
        self.w = w.float() if w is not None else torch.ones((x.shape[0], 1), dtype=torch.float32)
        assert self.w.shape == (x.shape[0], 1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.w[idx]


def create_dataloader(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor | None = None,
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    assert batch_size > 0
    assert num_workers >= 0
    dataset = OptionDataset(x, y, w)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def build_dataloaders(
    x: torch.Tensor,
    y: torch.Tensor,
    train_split: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    assert 0.0 < train_split < 1.0
    assert batch_size > 0
    assert num_workers >= 0

    dataset = OptionDataset(x, y)
    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train
    assert n_train > 0
    assert n_val > 0

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    return train_loader, val_loader
