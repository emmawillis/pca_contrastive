import random
from src.builder import create_model
import torch 
import pandas as pd
import h5py
import argparse
from sklearn.metrics import cohen_kappa_score
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class SlideBagOfEncodingsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings_root, split_csv):
       self.encodings_root = encodings_root
       self.split_csv = split_csv
       self.labels = pd.read_csv(split_csv)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        sid = row.get('FILENAME')
        isup = row.get('isup_grade')

        with h5py.File(f"{self.encodings_root}/{sid}_{isup}.h5", "r") as file:
            bag_of_encodings = file['features'][...].squeeze(0)
            return torch.FloatTensor(bag_of_encodings), int(isup), sid

def run_epoch(dataloader, model, criterion, optimizer):
    model.train()  
    for bag, y, sid in dataloader:
        bag = bag.to(device)                  # shape [1, N, 1536]
        y   = y.to(device).long()             # shape [1]
        out, encodings = model(bag)
        logits = out['logits']
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

def validate_epoch(dataloader, model, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    per_class_correct = None
    per_class_total   = None

    all_y, all_pred = [], []

    with torch.no_grad():
        for bag, y, sid in dataloader:
            bag = bag.to(device)           # [1, N, 1536]
            y   = y.to(device).long()      # [1]
            out, _ = model(bag)
            logits = out['logits']         # [1, C]
            loss = criterion(logits, y)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)   # [1]
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            all_y.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

            # init per-class trackers once we know C
            if per_class_correct is None:
                C = logits.size(-1)
                per_class_correct = [0] * C
                per_class_total   = [0] * C

            # update per-class counts (works even if batch_size=1)
            y_cpu = y.view(-1).cpu()
            p_cpu = preds.view(-1).cpu()
            
            for c in range(len(per_class_total)):
                mask = (y_cpu == c)
                n_c  = int(mask.sum().item())
                if n_c:
                    per_class_total[c]   += n_c
                    per_class_correct[c] += int((p_cpu[mask] == c).sum().item())

    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = total_correct / max(1, total_samples)

    # compute per-class accuracies (skip classes with 0 samples)
    per_class_acc = {
        c: (per_class_correct[c] / per_class_total[c])
        for c in range(len(per_class_total)) if per_class_total[c] > 0
    }
    balanced_acc = sum(per_class_acc.values()) / max(1, len(per_class_acc))

    qwk = cohen_kappa_score(all_y, all_pred, weights="quadratic")

    print(f"Validation Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | Balanced Acc: {balanced_acc:.4f} | QWK: {qwk:.4f}")
    print("Per-class Acc:", {f"ISUP {c}": f"{acc:.3f}" for c, acc in per_class_acc.items()})

    model.train()
    return qwk


def main():
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--encodings_root', type=str, required=True, help='Path to the root directory containing the encodings.')
    parser.add_argument('--split_csvs_root', type=str, required=True, help='Path to the CSV parent directory')
    parser.add_argument('--ckpt_path', type=str, default='best_model.pth')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--proj_dim', type=int, default=512)
    
    args = parser.parse_args()

    #  with HuggingFace's AutoModel using from_pretrained
    # https://huggingface.co/mahmoodlab/abmil.base.uni_v2.pc108-24k
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
    if args.proj_dim != 512:
        model.model.classifier = ABMILHead1024(in_dim=512, emb_dim=args.proj_dim, num_classes=num_classes).to(device)

    model.to(device)

    print(model)

    train_dataset = SlideBagOfEncodingsDataset(encodings_root=args.encodings_root, split_csv=args.split_csvs_root + '/train.csv')
    val_dataset = SlideBagOfEncodingsDataset(encodings_root=args.encodings_root, split_csv=args.split_csvs_root + '/val.csv')
    test_dataset = SlideBagOfEncodingsDataset(encodings_root=args.encodings_root, split_csv=args.split_csvs_root + '/test.csv')
    # item, label = train_dataset.__getitem__(0)
    # print(item.shape, label)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # bag, y, sid = next(iter(train_dataloader))
    # print(bag.shape, bag.dtype, y, sid)        # expect: [N,1536], float32, int, str
    # logits = model(bag.to(device))              # expect: [1, 6]
    # print(logits)

    # Training loops:
    model.train()

    # 1) Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2) Unfreeze only the classifier head (for a warm-up phase)
    for p in model.model.classifier.parameters():
        p.requires_grad = True

    # 3) Optimize the head only
    optimizer = torch.optim.AdamW(model.model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_qwk = -1

    for epoch in range(2):  # warm-up for 2 epochs
        run_epoch(train_dataloader, model, criterion, optimizer)
        print(f"Epoch {epoch+1}/2 completed.")
        qwk = validate_epoch(val_dataloader, model, criterion)
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save({
                "epoch": epoch,
                "best_qwk": best_qwk,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "arch": {"head": f"mlp{args.proj_dim}", "num_classes": num_classes}
            }, args.ckpt_path)

            print("Best model saved.")

    # Unfreeze everything for remaining epochs
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(args.epochs - 2):
        run_epoch(train_dataloader, model, criterion, optimizer)
        print(f"Epoch {epoch+3}/{args.epochs} completed.")
        qwk = validate_epoch(val_dataloader, model, criterion)
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save({
                "epoch": epoch,
                "best_qwk": best_qwk,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, args.ckpt_path)

            print("Best model saved.")


if __name__ == "__main__":
    main()
