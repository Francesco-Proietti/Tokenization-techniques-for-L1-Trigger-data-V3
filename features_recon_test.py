#!/usr/bin/env python3
"""First test script""" 
# At the moment it works only for MLP VQ-VAE and jet-constituents dataloader

# Import libraries
from pathlib import Path

import lightning as pl
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import models and data-loadings
from src.data.event_jets_data_loading import EventJetsL1TriggerDataset
from src.data.event_particles_data_loading import EventPartL1TriggerDataset
from src.data.jet_constituents_data_loading import JetConstL1TriggerDataset
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


def main():

    # Set seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Dataset not preprocessed (jet-constituents)
    dataset_not_prep = JetConstL1TriggerDataset(
        parquet_dirs="/run/media/francesco/STORAGE/data_cern/Test",
        max_particles=128,
        features=["L1T_PUPPIPart_PT",
                "L1T_PUPPIPart_Eta",
                "L1T_PUPPIPart_Phi",
                "L1T_PUPPIPart_PuppiW",
                "L1T_JetPuppiAK4_PT",
                "L1T_JetPuppiAK4_Eta",
                "L1T_JetPuppiAK4_Phi",
                "L1T_JetPuppiAK4_Mass",
                "L1T_JetPuppiAK4_ConstituentsIdx"
        ],
        preprocessing=False
    )

    # Dataset preprocessed (jet constituents)
    dataset_prep = JetConstL1TriggerDataset(
        parquet_dirs="/run/media/francesco/STORAGE/data_cern/Test",
        max_particles=128,
        features=["L1T_PUPPIPart_PT",
                "L1T_PUPPIPart_Eta",
                "L1T_PUPPIPart_Phi",
                "L1T_PUPPIPart_PuppiW",
                "L1T_JetPuppiAK4_PT",
                "L1T_JetPuppiAK4_Eta",
                "L1T_JetPuppiAK4_Phi",
                "L1T_JetPuppiAK4_Mass",
                "L1T_JetPuppiAK4_ConstituentsIdx"
        ],
        preprocessing=True
    )

    # Dataloader not preprocessed
    data_loader_not_prep = torch.utils.data.DataLoader(
        dataset_not_prep,
        batch_size=32,
        num_workers=0,
        pin_memory=True
    )

    # Dataloader preprocessed
    data_loader_prep = torch.utils.data.DataLoader(
        dataset_prep,
        batch_size=32,
        num_workers=0,
        pin_memory=True
    )

    # Checkpoint selection
    checkpoint = input("Enter the complete checkpoint's path: ")

    path = Path(checkpoint)

    # Model (uncomment the one you want to test)
    model = MLPVQVAE.load_from_checkpoint(path, weights_only=False)
    #model = TranformerVQVAE.load_from_checkpoint(path, weights_only=False)

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
    
    # Collect original features
    for batch in tqdm(data_loader_not_prep, desc="Looking for original features"):
        
        x, m, j = batch

        pt_o = x[:, :, 0]
        eta_o = x[:, :, 1]
        phi_o = x[:, :, 2]

        pt_orig.extend(pt_o[m].cpu())
        eta_orig.extend(eta_o[m].cpu())
        phi_orig.extend(phi_o[m].cpu())
    
    # Collect reconstructed features and codebook indices
    for batch in tqdm(data_loader_prep, desc="Computing reconstructed features"):

        x, m, j = batch

        x = x.to(device)
        m = m.to(device)
        j = j.to(device)

        with torch.no_grad():

            output = model(x,m)
        
            reco = inverse_preprocessing(output[0], m, j)
            
            pt_r = reco[:, :, 0]
            eta_r = reco[:, :, 1]
            phi_r = reco[:, :, 2]

            pt_reco.extend(pt_r[m].cpu())
            eta_reco.extend(eta_r[m].cpu())
            phi_reco.extend(phi_r[m].cpu())

            idx.extend(output[2].flatten().cpu())

    print("Plotting...")
    
    # Hyperparameters
    ckpt = torch.load(path, weights_only=False)
    model_name = str(ckpt["hyper_parameters"]["cfg"]["name"])
    cb_size = str(ckpt["hyper_parameters"]["cfg"]["codebook_size"])
    rot = str(ckpt["hyper_parameters"]["cfg"]["rotation_trick"])
    
    # Codebook usage
    cb_usage = len(torch.unique(torch.stack(idx))) / int(cb_size)
  
    # PT plot
    plt.hist(pt_orig, density=True, bins=50, color="blue", label="Original")
    plt.hist(pt_reco, density=True, bins=50, histtype="step", color="red", label="Reconstructed")
    plt.xlabel("PT [GeV]")
    plt.ylabel("Density")
    plt.title(model_name + " VQ-VAE, CB_size: " + cb_size + ", Rotation_trick: " + rot)
    plt.legend()
    plt.show()
    
    # Eta plot
    plt.hist(eta_orig, density=True, bins=50, color="blue", label="Original")
    plt.hist(eta_reco, density=True, bins=50, histtype="step", color="red", label="Reconstructed")
    plt.xlabel("Eta")
    plt.ylabel("Density")
    plt.title(model_name + " VQ-VAE, CB_size: " + cb_size + ", Rotation_trick: " + rot)
    plt.legend()
    plt.show()
    
    # Phi plot
    plt.hist(phi_orig, density=True, bins=50, color="blue", label="Original")
    plt.hist(phi_reco, density=True, bins=50, histtype="step", color="red", label="Reconstructed")
    plt.xlabel("Phi")
    plt.ylabel("Density")
    plt.title(model_name + " VQ-VAE, CB_size: " + cb_size + ", Rotation_trick: " + rot)
    plt.legend()
    plt.show()
    
    # Codebook usage plot
    plt.hist(idx, density=True, bins=50, color="orange")
    plt.xlim(0, 512)
    plt.xlabel(f"Quantization indices (CB-usage={cb_usage})")
    plt.ylabel("Density")
    plt.title(model_name + " VQ-VAE, CB_size: " + cb_size + ", Rotation_trick: " + rot)
    plt.show()

    print("Done!")


if __name__ == "__main__":
    main()