import os
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

from monai.networks.nets import SegResNet
from monai.inferers import SlidingWindowInferer


# -----------------------------
# 1️⃣ WRAPPER FUNCTION
# -----------------------------
def run_inference(t1_path, t1ce_path, t2_path, flair_path, ckpt_path):
    print(f"🚀 Starting inference with checkpoint: {ckpt_path}")
    
    # -----------------------------
    # 2️⃣ LOAD & STACK MODALITIES
    # -----------------------------
    t1    = nib.load(t1_path).get_fdata()
    t1ce  = nib.load(t1ce_path).get_fdata()
    t2    = nib.load(t2_path).get_fdata()
    flair = nib.load(flair_path).get_fdata()

    # Stack → (4, H, W, D)
    img = np.stack([t1ce, t1, t2, flair], axis=0)

    image = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    print("✅ Input tensor:", image.shape)

    # -----------------------------
    # 3️⃣ Z-SCORE NORMALIZATION (CRITICAL)
    # -----------------------------
    for c in range(image.shape[1]):
        channel = image[0, c]
        mask = channel != 0
        if mask.any():
             channel[mask] = (channel[mask] - channel[mask].mean()) / (channel[mask].std() + 1e-8)
        image[0, c] = channel

    # -----------------------------
    # 4️⃣ PAD DEPTH (155 → 160)
    # -----------------------------
    orig_shape = image.shape
    _, _, H, W, D = image.shape
    if D < 160:
        image = torch.nn.functional.pad(image, (0, 160 - D, 0, 0, 0, 0))

    print("✅ After padding:", image.shape)

    # -----------------------------
    # 5️⃣ LOAD MODEL
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        init_filters=16,
        blocks_down=(1,2,2,4),
        blocks_up=(1,1,1),
        dropout_prob=0.2
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    image = image.to(device)

    # -----------------------------
    # 6️⃣ INFERENCE (MONAI)
    # -----------------------------
    inferer = SlidingWindowInferer(
        roi_size=(240,240,160),
        sw_batch_size=1,
        overlap=0.5
    )

    with torch.no_grad():
        logits = inferer(image, model)
        probs = torch.sigmoid(logits)

    # -----------------------------
    # 7️⃣ BraTS POST-PROCESSING
    # Channels → Labels
    # 0=TC → 1
    # 1=WT → 2
    # 2=ET → 4
    # -----------------------------
    binary = (probs > 0.5).int()

    seg = torch.where(
        binary[:, 2] == 1, 4,
        torch.where(binary[:, 0] == 1, 1,
                    torch.where(binary[:, 1] == 1, 2, 0))
    )

    seg = seg[..., :D]   # remove padding
    seg_np = seg.squeeze(0).cpu().numpy().astype(np.uint8)

    print("✅ Unique labels:", np.unique(seg_np))

    # -----------------------------
    # 8️⃣ SAVE SEGMENTATION
    # -----------------------------
    # Create a temporary path for output or use a fixed one
    output_path = "BRATS_prediction.nii.gz"
    
    ref_nii = nib.load(flair_path)
    out_nii = nib.Nifti1Image(seg_np, ref_nii.affine, ref_nii.header)
    nib.save(out_nii, output_path)
    print("✅ Saved prediction to:", output_path)

    # -----------------------------
    # 9️⃣ CORRECT VOLUME COMPUTATION
    # -----------------------------
    voxel_volume = np.prod(ref_nii.header.get_zooms()[:3]) / 1000.0

    WT = np.sum(np.isin(seg_np, [1,2,4])) * voxel_volume
    TC = np.sum(np.isin(seg_np, [1,4])) * voxel_volume
    ET = np.sum(seg_np == 4) * voxel_volume

    volumes = {
        "Whole Tumor (WT)": f"{WT:.2f} cm³",
        "Tumor Core (TC)": f"{TC:.2f} cm³",
        "Enhancing Tumor (ET)": f"{ET:.2f} cm³"
    }

    print("\n📊 Tumor Volumes (cm³)")
    print(f"WT: {WT:.2f}")
    print(f"TC: {TC:.2f}")
    print(f"ET: {ET:.2f}")

    # -----------------------------
    # 🔟 BEST TUMOR SLICE (CALCULATION ONLY)
    # -----------------------------
    tumor_mask = np.isin(seg_np, [1,2,4])
    if tumor_mask.any():
        best_z = np.argmax(np.sum(tumor_mask, axis=(0,1)))
    else:
        best_z = seg_np.shape[2] // 2 
    print(f"🧠 Best tumor slice: {best_z}")

    # Return raw data for frontend visualization
    return output_path, volumes, flair, seg_np, int(best_z)

# Allow standalone execution if needed (optional, keeping original behavior via main check could be done but user asked for function)
if __name__ == "__main__":
    # Example usage if run directly (assumes paths exist)
    pass
