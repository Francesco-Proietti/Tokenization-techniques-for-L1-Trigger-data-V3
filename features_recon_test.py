#!/usr/bin/env python3
"""First test script"""

# Import libraries
import hydra
from omegaconf import DictConfig, OmegaConf

from pathlib import Path

import lightning as pl
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import model and data registries
from src.models.model_registry import MODEL_REGISTRY
from src.data.data_registry import DATA_REGISTRY
from src.models.mlp_vqvae import MLPVQVAE
from src.models.transformer_vqvae import TransformerVQVAE

# Inverse preprocessing function
def inverse_preprocessing(
    feats: torch.Tensor,
    mask: torch.Tensor,
    jet_feats: torch.Tensor,
):
    """
    Inverse preprocessing for constituent features.

    Args:
        feats:
            Tensor of shape [B, N, 3]
            containing:
                [:,:,0] -> preprocessed pt
                [:,:,1] -> preprocessed eta
                [:,:,2] -> preprocessed phi
        
        mask:
            Tensor of shape [B, N]
            containing True entris for valid constituents

        jet_feats:
            Tensor of shape [B, 4]
            containing jet features.

    Returns:
        Tensor of shape [B, N, 3]
        with original:
            pt, eta, phi
    """

    jet_eta = jet_feats[:, 1]
    jet_phi = jet_feats[:, 2]

    out = feats.clone()

    mask3d = mask.unsqueeze(-1)  # [B, N, 1]

    # PT inverse
    pt = torch.exp(out[:, :, 0] + 1.8) - 1e-8

    # Eta inverse-
    eta = out[:, :, 1] * 3.0
    eta += jet_eta[:, None]

    # Phi inverse
    phi = out[:, :, 2] * 3.0
    # wrap phi to [-pi, pi]
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    phi += jet_phi[:, None]

    recovered = torch.stack([pt, eta, phi], dim=-1)
    
    # Keep padding entries to zero
    recovered = torch.where(
        mask3d,
        recovered,
        torch.zeros_like(recovered),
    )
    
    return recovered

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config"
)
def main(cfg: DictConfig):

    # Set seed for reproducibility
    pl.seed_everything(cfg.trainer.seed, workers=True)
    
    # Set custom config (only for this script)
    dm_not_prep_cfg = OmegaConf.merge(
        cfg.data,
        {
            "preprocessing": False,
        },
    )

    dm_prep_cfg = OmegaConf.merge(
        cfg.data,
        {
            "preprocessing": True,
        },
    )

    # DataModule
    data_name = cfg.data.name
    DataModuleClass = DATA_REGISTRY[data_name]

    data_module_not_prep = DataModuleClass(dm_not_prep_cfg, batch_size=cfg.trainer.batch_size)
    data_module_prep = DataModuleClass(dm_prep_cfg, batch_size=cfg.trainer.batch_size)
    
    # Dataloader initialization
    data_loader_not_prep = data_module_not_prep.test_dataloader()
    data_loader_prep = data_module_prep.test_dataloader()

    # Checkpoint selection
    checkpoint = input("Enter the complete checkpoint's path: ")

    path = Path(checkpoint)

    # Model
    model_name = cfg.model.name
    
    if model_name == "mlp":
        model = MLPVQVAE.load_from_checkpoint(path, weights_only=False)
    elif model_name == "transformer":
        model = TransformerVQVAE.load_from_checkpoint(path, weights_only=False)

    # Set model in evaluation mode
    model.eval()
    
    # Select GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    pt_orig = []
    eta_orig = []
    phi_orig = []
    
    pt_reco = []
    eta_reco = []
    phi_reco = []

    idx = []

    for batch in tqdm(data_loader_not_prep, desc="Looking for original features"):
        
        x, m, j = batch

        pt_o = x[:, :, 0]
        eta_o = x[:, :, 1]
        phi_o = x[:, :, 2]

        pt_orig.extend(pt_o[m].cpu())
        eta_orig.extend(eta_o[m].cpu())
        phi_orig.extend(phi_o[m].cpu())
    
    for batch in tqdm(data_loader_prep, desc="Computing reconstructed features"):

        x, m, j = batch

        x = x.to(device)
        m = m.to(device)
        j = j.to(device)

        with torch.no_grad():

            output = model(x,m)

            pt_out = output[0][:, :, 0]
        
            reco = inverse_preprocessing(output[0], m, j)
            
            pt_r = reco[:, :, 0]
            eta_r = reco[:, :, 1]
            phi_r = reco[:, :, 2]

            pt_reco.extend(pt_r[m].cpu())
            eta_reco.extend(eta_r[m].cpu())
            phi_reco.extend(phi_r[m].cpu())

            idx.extend(output[2].flatten().cpu())

    print("Plotting...")
    
    ckpt = torch.load(path, weights_only=False)
    model_name = str(ckpt["hyper_parameters"]["cfg"]["name"])
    cb_size = str(ckpt["hyper_parameters"]["cfg"]["codebook_size"])
    rot = str(ckpt["hyper_parameters"]["cfg"]["rotation_trick"])
    
    cb_usage = len(torch.unique(torch.stack(idx))) / int(cb_size)

    plt.hist(pt_orig, density=True, bins=50, color="blue", label="Original")
    plt.hist(pt_reco, density=True, bins=50, histtype="step", color="red", label="Reconstructed")
    plt.xlabel("PT [GeV]")
    plt.ylabel("Density")
    plt.title(model_name + " VQ-VAE, CB_size: " + cb_size + ", Rotation_trick: " + rot)
    plt.legend()
    plt.show()

    plt.hist(eta_orig, density=True, bins=50, color="blue", label="Original")
    plt.hist(eta_reco, density=True, bins=50, histtype="step", color="red", label="Reconstructed")
    plt.xlabel("Eta")
    plt.ylabel("Density")
    plt.title(model_name + " VQ-VAE, CB_size: " + cb_size + ", Rotation_trick: " + rot)
    plt.legend()
    plt.show()
    
    plt.hist(phi_orig, density=True, bins=50, color="blue", label="Original")
    plt.hist(phi_reco, density=True, bins=50, histtype="step", color="red", label="Reconstructed")
    plt.xlabel("Phi")
    plt.ylabel("Density")
    plt.title(model_name + " VQ-VAE, CB_size: " + cb_size + ", Rotation_trick: " + rot)
    plt.legend()
    plt.show()

    plt.hist(idx, density=True, bins=50, color="orange")
    plt.xlim(0, 512)
    plt.xlabel(f"Quantization indices (CB-usage={cb_usage})")
    plt.ylabel("Density")
    plt.title(model_name + " VQ-VAE, CB_size: " + cb_size + ", Rotation_trick: " + rot)
    plt.show()

    print("Done!")


if __name__ == "__main__":
    main()