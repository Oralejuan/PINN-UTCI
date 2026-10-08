"""Entrenamiento por etapas con weighted loss y métricas honestas."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count()))

from config.settings import PROJECT_ROOT
from src.pinn.dataset import load_datasets
from src.pinn.losses import total_loss
from src.pinn.model import UTCIMLP

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def _evaluate(model, loader, device):
    """MAE y MSE no ponderados (reflejan la distribución real de los datos)."""
    model.eval()
    mae_sum, mse_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            Xb, yb = batch[0], batch[1]
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            err = pred - yb
            mae_sum += err.abs().sum().item()
            mse_sum += (err ** 2).sum().item()
            n += yb.numel()
    model.train()
    return mae_sum / n, mse_sum / n


def train_pinn(epochs=500, batch_size=16384, lr=1e-3, n_samples=200_000,
               split_mode="random"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count())
    print(f"[train] device={device} | threads={torch.get_num_threads()}")

    print(f"[train] split_mode = {split_mode}")
    ds_train, ds_val, ds_test = load_datasets(
        n_samples=n_samples, split_mode=split_mode
    )
    print(f"[train] train={len(ds_train):,} | val={len(ds_val):,} | test={len(ds_test):,}")

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)
    loader_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False)

    in_dim = ds_train.X.shape[1]
    model = UTCIMLP(in_dim=in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    mean = torch.from_numpy(ds_train.mean).float().to(device)
    std  = torch.from_numpy(ds_train.std).float().to(device)

    # ── Etapa 1: solo datos ──
    print("\n=== Etapa 1: solo datos (weighted) ===")
    for epoch in range(epochs // 2):
        model.train()
        for Xb, yb, wb in loader_train:
            Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            L, _ = total_loss(model, Xb, yb, wb, mean, std,
                              lambda_data=1.0, lambda_phys=0.0, lambda_smooth=0.0)
            L.backward()
            opt.step()

        if (epoch + 1) % 10 == 0:
            mae, mse = _evaluate(model, loader_val, device)
            print(f"  epoch {epoch+1:3d} | val_MAE={mae:.3f} °C | val_RMSE={np.sqrt(mse):.3f} °C")

    ckpt_stage1 = MODELS_DIR / "pinn_utci_aburra_stage1.pt"
    torch.save({
        "model_state": model.state_dict(),
        "in_dim": in_dim,
        "feature_mean": ds_train.mean,
        "feature_std": ds_train.std,
    }, ckpt_stage1)
    print(f"[train] checkpoint Etapa 1 → {ckpt_stage1}")

    # ── Etapa 2: datos + física ──
    print("\n=== Etapa 2: data + fisica (weighted) ===")
    for epoch in range(epochs // 2):
        lambda_phys = 0.5 * (epoch / max(epochs // 2 - 1, 1))
        model.train()
        for Xb, yb, wb in loader_train:
            Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            L, parts = total_loss(model, Xb, yb, wb, mean, std,
                                   lambda_data=1.0,
                                   lambda_phys=lambda_phys,
                                   lambda_smooth=0.05)
            L.backward()
            opt.step()

        if (epoch + 1) % 10 == 0:
            mae, mse = _evaluate(model, loader_val, device)
            print(f"  epoch {epoch+1:3d} | val_MAE={mae:.3f} | val_RMSE={np.sqrt(mse):.3f} "
                  f"| L_phys={parts['phys']:.2e} | lambda={lambda_phys:.3f}")

    # ── Evaluación final ──
    print("\n=== Evaluación final sobre test (sin ponderar) ===")
    mae, mse = _evaluate(model, loader_test, device)
    rmse = float(np.sqrt(mse))

    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader_test:
            Xb, yb = batch[0], batch[1]
            preds.append(model(Xb.to(device)).cpu().numpy().ravel())
            trues.append(yb.numpy().ravel())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    ss_res = np.sum((preds - trues) ** 2)
    ss_tot = np.sum((trues - trues.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    print(f"  MAE  = {mae:.3f} °C")
    print(f"  RMSE = {rmse:.3f} °C")
    print(f"  R²   = {r2:.4f}")
    print(f"  N    = {len(trues):,}")

    ckpt = MODELS_DIR / "pinn_utci_aburra.pt"
    torch.save({
        "model_state": model.state_dict(),
        "in_dim": in_dim,
        "feature_mean": ds_train.mean,
        "feature_std": ds_train.std,
    }, ckpt)
    print(f"\n[train] modelo final → {ckpt}")

    metrics = {
        "test_mae": float(mae),
        "split_mode": split_mode,
        "test_rmse": rmse,
        "test_r2": r2,
        "n_train": len(ds_train),
        "n_val": len(ds_val),
        "n_test": len(ds_test),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_samples": n_samples,
        "weighted_loss": True,
    }
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[train] métricas → {metrics_path}")

    return model, (ds_train, ds_val, ds_test)


if __name__ == "__main__":
    train_pinn()