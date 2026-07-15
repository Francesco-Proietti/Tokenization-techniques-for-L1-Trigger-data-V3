import torch

tokens = torch.load("tokens.pt")
masks = torch.load("masks.pt")
labels = torch.load("labels.pt")

print("tokens")
print(" shape:", tokens.shape)
print(" dtype:", tokens.dtype)
print(" min:", tokens.min())
print(" max:", tokens.max())

print("\nmasks")
print(" shape:", masks.shape)
print(" dtype:", masks.dtype)
print(" unique:", torch.unique(masks))

print("\nlabels")
print(" shape:", labels.shape)
print(" dtype:", labels.dtype)
print(" unique:", torch.unique(labels))