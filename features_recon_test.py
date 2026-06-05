import sys
from pathlib import Path

sys.path.append(str(Path().resolve().parent))

from data.jet_constituents_data_loading import L1TriggerDataset, L1TriggerDataModule
from src.models.mlp_vqvae import MLPVQVAE
from src.models.transformer_vqvae import TransformerVQVAE

import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

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


def jet_mass(pt, eta, phi, mask):

    mask = mask.float()

    px = pt * torch.cos(phi) * mask
    py = pt * torch.sin(phi) * mask
    pz = pt * torch.sinh(eta) * mask

    E = pt * torch.cosh(eta) * mask

    px_tot = px.sum(dim=-1)
    py_tot = py.sum(dim=-1)
    pz_tot = pz.sum(dim=-1)
    E_tot  = E.sum(dim=-1)

    m2 = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2

    return torch.sqrt(torch.clamp(m2, min=0))


def main():
    
    # Datamodule not preprocessed
    datamodule_not_prep = L1TriggerDataModule(
        parquet_dirs_train="/run/media/francesco/STORAGE/data_cern/Train",
        parquet_dirs_val="/run/media/francesco/STORAGE/data_cern/Val",
        parquet_dirs_test="/run/media/francesco/STORAGE/data_cern/Test",
        max_particles=128,
        batch_size=32,
        num_workers=0,
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
    
    # Datamodule preprocessed
    datamodule_prep = L1TriggerDataModule(
        parquet_dirs_train="/run/media/francesco/STORAGE/data_cern/Train",
        parquet_dirs_val="/run/media/francesco/STORAGE/data_cern/Val",
        parquet_dirs_test="/run/media/francesco/STORAGE/data_cern/Test",
        max_particles=128,
        batch_size=32,
        num_workers=0,
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
    
    # Print to terminal the possible choices for the model
    print("1) MLP_VQVAE")
    print("2) Transformer_VQVAE")
    
    choice = input("Select 1 or 2: ")
    
    while choice not in (["1","2"]):
        choice = input("INVALID ENTRY! Select 1 or 2: ")
    
    # Checkpoint selection
    checkpoint = input("Enter the complete checkpoint's path: ")
    
    path = Path(checkpoint)
    
    if choice == "1":
        model = MLPVQVAE.load_from_checkpoint(path, weights_only=False)
    elif choice == "2":
        model = TransformerVQVAE.load_from_checkpoint(path, weights_only=False)

    print("Wait until the process is completed")
    
    # Dataloaders initialization
    dataloader_test_not_prep = datamodule_not_prep.test_dataloader()
    dataloader_test_prep = datamodule_prep.test_dataloader()
    
    # Set model in evaluation mode
    model.eval()
    
    # Select GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    pt_orig = []
    eta_orig = []
    phi_orig = []

    pt_output = []
    pt_orig_no_pro = []
    
    pt_reco = []
    eta_reco = []
    phi_reco = []

    j_m_reco = []

    idx = []

    j_m_orig = []
    
    for batch in tqdm(dataloader_test_prep, desc="Looking for original features"):
        
        x, m, j = batch

        pt_o_no_pro = x[:, :, 0]
        
        pt_orig_no_pro.extend(pt_o_no_pro[m].cpu())

    for batch in tqdm(dataloader_test_not_prep, desc="Looking for original features"):
        
        x, m, j = batch

        pt_o = x[:, :, 0]
        eta_o = x[:, :, 1]
        phi_o = x[:, :, 2]

        j_m_o = j[:,3]

        pt_orig.extend(pt_o[m].cpu())
        eta_orig.extend(eta_o[m].cpu())
        phi_orig.extend(phi_o[m].cpu())

        j_m_orig.extend(j_m_o.flatten().cpu())
    

    for batch in tqdm(dataloader_test_prep, desc="Computing reconstructed features"):

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

            j_m_r = jet_mass(pt_r, eta_r, phi_r, m)

            pt_reco.extend(pt_r[m].cpu())
            eta_reco.extend(eta_r[m].cpu())
            phi_reco.extend(phi_r[m].cpu())

            j_m_reco.extend(j_m_r.flatten().cpu())

            idx.extend(output[2].flatten().cpu())

            pt_output.extend(pt_out[m].cpu())

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

    plt.hist(np.array(pt_reco) - np.array(pt_orig), density=True, bins=50, color="blue", label="Residuals")
    plt.xlabel("PT [GeV]")
    plt.ylabel("Density")
    plt.xlim(-100,100)
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

    plt.hist(pt_orig_no_pro, density=True, bins=50, color="blue", label="Original")
    plt.hist(pt_output, density=True, bins=50, histtype="step", color="red", label="Reconstructed")
    plt.xlabel("PT (preprocessed)")
    plt.ylabel("Density")
    plt.title(model_name + " VQ-VAE, CB_size: " + cb_size + ", Rotation_trick: " + rot)
    plt.legend()
    plt.show()

    plt.hist(j_m_orig, density=True, bins=50, color="blue", label="Original")
    plt.hist(j_m_reco, density=True, bins=50, color="red", label="Reconstructed")
    plt.xlabel("Jet mass")
    plt.ylabel("Density")
    plt.title(model_name + " VQ-VAE, CB_size: " + cb_size + ", Rotation_trick: " + rot)
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