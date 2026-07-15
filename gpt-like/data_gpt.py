import torch

class TokenDataset(Dataset):

    def __init__(self, token_path, mask_path, label_path=None):

        self.tokens = torch.load(token_path)
        self.masks = torch.load(mask_path)

        self.labels = None

        if label_path is not None:
            self.labels = torch.load(label_path)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):

        sample = {
            "tokens": self.tokens[idx],
            "mask": self.masks[idx],
        }

        if self.labels is not None:
            sample["label"] = self.labels[idx]

        return sample