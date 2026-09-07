from pathlib import Path
import numpy as np
import torch
from src.builder import create_model
from train import SlideBagOfEncodingsDataset
import torch.nn as nn

# pick device once
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

def export_bag_embeddings(dataloader, ckpt_path, outdir: Path, proj_dim=512):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # rebuild the same arch you trained (stock 512->C head here)
    model = create_model('abmil.base.uni_v2.pc108-24k', from_pretrained=True, num_classes=6) # 5 ISUP grades + benign

    class ABMILHead1024(nn.Module):
        def __init__(self, in_dim=512, emb_dim=1024, num_classes=6):
            super().__init__()
            self.proj = nn.Linear(in_dim, emb_dim)   # 512 -> 1024
            self.act  = nn.ReLU()
            self.norm = nn.LayerNorm(emb_dim)
            self.cls  = nn.Linear(emb_dim, num_classes)  # 1024 -> C

        def forward(self, z):          # z is the pooled bag vector from ABMIL, shape [B, 512]
            z1024 = self.norm(self.act(self.proj(z)))   # [B, 1024]
            logits = self.cls(z1024)                     # [B, C]
            return logits

    num_classes = 6
    if proj_dim != 512:
        model.model.classifier = ABMILHead1024(in_dim=512, emb_dim=proj_dim, num_classes=num_classes).to(device)

    model.to(device)
    ckpt = torch.load(ckpt_path, map_location="mps", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(model)

    # hook: capture classifier **input** (512-D bag vector)
    buf = {}
    def grab_bag(_, inp):
        buf["z"] = inp[0].detach()   # [B, 512] on current device
    h = model.model.classifier.cls.register_forward_pre_hook(grab_bag)     # -> [B,128]

    with torch.no_grad():
        for bag, y, sid in dataloader:
            # move input to same device as model
            bag = bag.to(device).float()              # expect [1, N, 1536]
            if bag.ndim == 4 and bag.size(0) == 1:    # safety if an extra dim slipped in
                bag = bag.squeeze(0)                  # -> [1, N, 1536]

            _ = model(bag)                            # triggers hook
            z = buf["z"].squeeze(0).cpu().numpy()     # -> (512,)

            # normalize sid to a string
            if isinstance(sid, (list, tuple)): sid = sid[0]
            if torch.is_tensor(sid): sid = sid.item()
            np.save(outdir / f"{sid}_abmil{proj_dim}.npy", z.astype(np.float32))

    h.remove()

from pathlib import Path
def export_all_splits(checkpoint, outdir, proj_dim):
    encodings_root = Path("/Users/emma/Desktop/QUEENS/THESIS/contrastive/histopathology_data/UNI2_panda_encodings_by_patch")
    split_csvs_root = Path("/Users/emma/Desktop/QUEENS/THESIS/contrastive/histopathology_model/MIL-Lab/panda_splits")

    train_dataset = SlideBagOfEncodingsDataset(encodings_root=encodings_root, split_csv=split_csvs_root / 'train.csv')
    val_dataset = SlideBagOfEncodingsDataset(encodings_root=encodings_root, split_csv=split_csvs_root / 'val.csv')
    test_dataset = SlideBagOfEncodingsDataset(encodings_root=encodings_root, split_csv=split_csvs_root / 'test.csv')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    export_bag_embeddings(train_dataloader, checkpoint, outdir / "train", proj_dim=proj_dim)
    export_bag_embeddings(val_dataloader,   checkpoint, outdir / "val", proj_dim=proj_dim)
    export_bag_embeddings(test_dataloader,  checkpoint, outdir / "test", proj_dim=proj_dim)



checkpoint768 = Path("/Users/emma/Desktop/QUEENS/THESIS/contrastive/histopathology_model/MIL-Lab/results/projected_768D/best_model.pth")
outdir768 = Path("/Users/emma/Desktop/QUEENS/THESIS/contrastive/histopathology_model/MIL-Lab/results/projected_768D/embeddings_768")
export_all_splits(checkpoint768, outdir768, proj_dim=768)
    
# checkpoint256 = Path("/Users/emma/Desktop/QUEENS/THESIS/contrastive/histopathology_model/MIL-Lab/results/projected_256D/best_model.pth")
# outdir256 = Path("/Users/emma/Desktop/QUEENS/THESIS/contrastive/histopathology_model/MIL-Lab/results/projected_256D/embeddings_256")
# export_all_splits(checkpoint256, outdir256, proj_dim=256)
