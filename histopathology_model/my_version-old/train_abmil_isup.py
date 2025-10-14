# train_abmil_isup_wandb.py
import argparse, os, math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from abmil_isup import ABMILISUP
from dataloader_panda_mil import make_loaders

def set_seed(seed: int = 1337):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    ys, yps = [], []
    for X, y, mask, _ in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits, _, _ = model(X, mask)
        y_hat = logits.argmax(dim=1)
        ys.append(y.detach().cpu()); yps.append(y_hat.detach().cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(yps).numpy()
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, total_count = 0.0, 0
    for X, y, mask, _ in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        logits, _, _ = model(X, mask)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # per-iteration cosine

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_count += bs
    return total_loss / max(1, total_count)

def main():
    p = argparse.ArgumentParser(description="Train gated ABMIL on PANDA (ISUP) with W&B logging")
    # data
    p.add_argument("--csv", required=True, type=str)
    p.add_argument("--root", required=True, type=str)
    p.add_argument("--fold", required=True, type=int)
    p.add_argument("--out", required=True, type=str)
    # training (UNI/UNI2 defaults)
    p.add_argument("--epochs", default=20, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--weight-decay", default=1e-4, type=float)
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--seed", default=1337, type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu", type=str)
    p.add_argument("--max-patches-train", default=None, type=int)
    p.add_argument("--use-weighted-sampler", action="store_true")
    # wandb
    p.add_argument("--wandb", default="disabled", choices=["disabled", "online", "offline"])
    p.add_argument("--wandb-project", default="panda-abmil")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", default="", help="comma-separated tags")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Data
    Ltrain, Lval, Ltest, *_ = make_loaders(
        csv_path=args.csv,
        root=args.root,
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_patches_train=args.max_patches_train,
        use_weighted_sampler=args.use_weighted_sampler,
        pin_memory=(device.type == "cuda"),
    )

    # Model (UNI Methods config)
    model = ABMILISUP(
        input_dim=1536, num_classes=6,
        proj_dim=512, attn_hidden=384,
        p_input_dropout=0.10, p_mid_dropout=0.25,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(Ltrain))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    # ---- Weights & Biases ----
    use_wandb = args.wandb != "disabled"
    if use_wandb:
        if args.wandb == "offline":
            os.environ["WANDB_MODE"] = "offline"
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()],
            config={
                **vars(args),
                "model": "Gated-ABMIL",
                "proj_dim": 512, "attn_hidden": 384,
                "p_input_dropout": 0.10, "p_mid_dropout": 0.25,
            },
        )
        wandb.watch(model, log="all", log_freq=100)

    # Train with early stopping on val loss
    best_val_loss, best_epoch, best_state = float("inf"), -1, None
    patience = 5

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, Ltrain, optimizer, scheduler, device)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, count = 0.0, 0
            for X, y, mask, _ in Lval:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                logits, _, _ = model(X, mask)
                loss = F.cross_entropy(logits, y)
                bs = y.size(0)
                val_loss += loss.item() * bs; count += bs
            val_loss /= max(1, count)

        val_metrics = evaluate(model, Lval, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} acc={val_metrics['acc']:.4f} "
              f"f1m={val_metrics['f1_macro']:.4f} qwk={val_metrics['qwk']:.4f} lr={lr_now:.6f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/acc": val_metrics["acc"],
                "val/f1_macro": val_metrics["f1_macro"],
                "val/qwk": val_metrics["qwk"],
                "lr": lr_now,
            }, step=epoch)

        # Early stopping (val loss)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": vars(args),
            }
        elif epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch} (no val loss improvement for {patience} epochs).")
            break

    # Save best checkpoint
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else {"model": model.state_dict()}, out_path)
    print(f"Saved best checkpoint (epoch {best_epoch}) to: {out_path}")

    # Final eval (val/test) from best
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    val_final = evaluate(model, Lval, device)
    print(f"[VAL FINAL] acc={val_final['acc']:.4f} f1m={val_final['f1_macro']:.4f} qwk={val_final['qwk']:.4f}")
    if use_wandb:
        import wandb
        wandb.log({
            "final/val/acc": val_final["acc"],
            "final/val/f1_macro": val_final["f1_macro"],
            "final/val/qwk": val_final["qwk"],
            "final/best_epoch": best_epoch,
        })

    if Ltest is not None:
        test_final = evaluate(model, Ltest, device)
        print(f"[TEST] acc={test_final['acc']:.4f} f1m={test_final['f1_macro']:.4f} qwk={test_final['qwk']:.4f}")
        if use_wandb:
            import wandb
            wandb.log({
                "final/test/acc": test_final["acc"],
                "final/test/f1_macro": test_final["f1_macro"],
                "final/test/qwk": test_final["qwk"],
            })
        # (optional) log checkpoint as artifact
        if use_wandb:
            artifact = wandb.Artifact("abmil-uni2-panda", type="model")
            artifact.add_file(str(out_path))
            wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()
