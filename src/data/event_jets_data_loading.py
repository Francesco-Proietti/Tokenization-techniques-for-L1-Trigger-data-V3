"""
Data-loading Implementation

It consists of an IterableDataset and a Lightning DataModule
"""

from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset, get_worker_info
import lightning as pl


class EventJetsL1TriggerDataset(IterableDataset):
    """
    IterableDataset for L1-trigger data from parquet files.

    Streams data lazily from parquet files instead of loading all into memory.
    Each event contains PUPPI particles with features: pT, eta, phi.
    """

    def __init__(
        self,
        parquet_dirs: List[str],
        max_jets: int = 8,
        features: List[str] = ["L1T_JetPuppiAK4_PT", "L1T_JetPuppiAK4_Eta", "L1T_JetPuppiAK4_Phi"],
        preprocessing: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            parquet_dirs: List of directories containing parquet files.
            max_jets: Maximum number of jets per event.
            features: List of feature to extract.
            preprocessing: Whether to apply preprocessing.
        """
        super().__init__()

        self.dataset = ds.dataset(parquet_dirs, format="parquet")
        self.max_jets = max_jets
        self.features = features
        self.preprocessing = preprocessing

    def _process_event(self, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single event row into padded features and mask.

        Returns:
            features: [max_jets, n_feats] tensor
            mask: [max_jets] boolean tensor
        """
        n_feats = len(self.features)
        feats = np.zeros((self.max_jets, n_feats), dtype=np.float32)
        mask = np.zeros(self.max_jets, dtype=bool)

        # Preprocessing 
        if self.preprocessing:
            
            pt = np.array(row["L1T_JetPuppiAK4_PT"])
            eta = np.array(row["L1T_JetPuppiAK4_Eta"])
            phi = np.array(row["L1T_JetPuppiAK4_Phi"])
            pt = np.log(pt + 1e-8) - 1.8  
            eta = eta / 3
            phi = phi / np.pi
            row["L1T_JetPuppiAK4_PT"] = pt
            row["L1T_JetPuppiAK4_Eta"] = eta
            row["L1T_JetPuppiAK4_Phi"] = phi

        # Padding
        for feat_idx, feat_name in enumerate(self.features):
            jets_feat = row[feat_name]
            n_jets = min(len(jets_feat), self.max_jets)
            feats[:n_jets, feat_idx] = jets_feat[:n_jets]
            mask[:n_jets] = True
        
        return torch.FloatTensor(feats), torch.BoolTensor(mask)

    def __iter__(self) -> Iterator[Tuple]:
        """
        Iterate over all events using pyarrow.dataset scanner.
        """

        worker_info = get_worker_info()

        files = self.dataset.files

        if worker_info is None:
            assigned_files = files
        else:
            assigned_files = files[worker_info.id::worker_info.num_workers]

        dataset = ds.dataset(assigned_files, format="parquet")

        scanner = dataset.scanner(
            columns=self.features,
            use_threads=True,
        )

        for batch in scanner.to_batches():
            df = batch.to_pandas()

            for i in range(len(df)):
                yield self._process_event(df.iloc[i])


class EventJetsL1TriggerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for L1-trigger data.
    """

    def __init__(
        self,
        cfg,
        batch_size: int = 32
    ):
        """
        Initialize the DataModule.

        Args:
            parquet_dirs_train: Directories containing training parquet files.
            parquet_dirs_val: Directories containing validation data.
            parquet_dirs_test: Directories containing test data.
            max_jets: Maximum jets per event.
            batch_size: Batch size for dataloaders.
            num_workers: Workers for dataloaders.
            features: Features to extract.
            preprocessing: Whether to apply preprocessing.
        """
        super().__init__()

        self.train_dirs = cfg.train_path
        self.val_dirs = cfg.val_path or []
        self.test_dirs = cfg.test_path or []
        self.max_jets = cfg.max_jets
        self.batch_size = batch_size
        self.num_workers = cfg.num_workers
        self.features = list(cfg.features)
        self.preprocessing = cfg.preprocessing

    def train_dataloader(self):
        """Return training dataloader."""
        self.train_dataset = EventJetsL1TriggerDataset(
            parquet_dirs=self.train_dirs,
            max_jets=self.max_jets,
            features=self.features,
            preprocessing=self.preprocessing
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
            #drop_last=True,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        self.val_dataset = EventJetsL1TriggerDataset(
            parquet_dirs=self.val_dirs,
            max_jets=self.max_jets,
            features=self.features,
            preprocessing=self.preprocessing
        )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            #drop_last=True,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        self.test_dataset = EventJetsL1TriggerDataset(
            parquet_dirs=self.test_dirs,
            max_jets=self.max_jets,
            features=self.features,
            preprocessing=self.preprocessing
        )
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            #drop_last=True,
        )