from typing import List
from rl4lms.data_pools.text_generation_pool import Sample
from torch.utils.data import Dataset, DataLoader


class TextGenDataset(Dataset):
    def __init__(self, samples: List[Sample]) -> None:
        super().__init__()
        self._samples = samples
        self._size = len(samples)

    def __len__(self):
        return self._size

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


def create_dataloader(samples: List[Sample], batch_size: int):
    dataset = TextGenDataset(samples)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return dataloader
