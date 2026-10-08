"""Evaluación de la PINN entrenada.

Carga el checkpoint, predice UTCI sobre las 4 temporadas y genera:
- Métricas globales (MAE, RMSE, R²)
- Métricas por temporada y por hora del día
- Mapas de UTCI real vs predicho (guardados como PNG)
"""

import numpy as np
import torch
import xarray as xr

from config.settings import PATHS, PROJECT_ROOT
from src.pinn.dataset import FEATURE_COLS, TARGET_COL
from src.pinn.model import UTCIMLP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UTCIMLP(in_dim=ckpt["in_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model, ckpt["feature_mean"], ckpt["feature_std"]


def predict_season(model, ckpt_mean, ckpt_std, season, device="cpu"):
    """Predice UTCI para toda una temporada, procesando por hora (bajo uso de RAM)."""
    nc_path = PATHS["processed"] / f"aburra_{season}_utci_local.nc"
    ds = xr.open_dataset(nc_path)

    n_h = ds.sizes["hour_of_day"]
    n_x = ds.sizes["x"]
    n_y = ds.sizes["y"]
    n_feats = len(FEATURE_COLS)

    x_vals = ds["x"].values.astype(np.float32)
    y_vals = ds["y"].values.astype(np.float32)
    h_vals = ds["hour_of_day"].values.astype(np.float32)
    season_id = {"dry": 0, "wet": 1, "dry2": 2, "wet2": 3}[season]

    # Rejillas 2D de coordenadas (n_y, n_x)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    # Convertir mean/std a tensores 1D
    mean_t = torch.from_numpy(np.asarray(ckpt_mean).reshape(-1)).float().to(device)
    std_t  = torch.from_numpy(np.asarray(ckpt_std).reshape(-1)).float().to(device)

    pred = np.full((n_h, n_y, n_x), np.nan, dtype=np.float32)
    real = np.full((n_h, n_y, n_x), np.nan, dtype=np.float32)

    with torch.no_grad():
        for h_idx in range(n_h):
            # 1) Features de esta hora
            feats = np.empty((n_feats, n_y, n_x), dtype=np.float32)
            for i, c in enumerate(FEATURE_COLS):
                if c in ds:
                    da = ds[c]
                    if "hour_of_day" in da.dims:
                        feats[i] = da.isel(hour_of_day=h_idx).values
                    else:
                        feats[i] = da.values
                else:
                    feats[i] = 0.0

            # 2) Target real
            real[h_idx] = ds[TARGET_COL].isel(hour_of_day=h_idx).values

            # 3) Máscara NaN
            mask = np.isfinite(feats).all(0) & np.isfinite(real[h_idx])
            if not mask.any():
                print(f"    hora {h_idx:2d}: sin píxeles válidos")
                continue

            # 4) Construir X para píxeles válidos → (N, 4 + n_feats)
            N = int(mask.sum())
            X = np.empty((N, 4 + n_feats), dtype=np.float32)
            X[:, 0] = X_grid[mask]
            X[:, 1] = Y_grid[mask]
            X[:, 2] = h_vals[h_idx]
            X[:, 3] = season_id
            X[:, 4:] = feats[:, mask].T

            # 5) Predecir en sub-lotes
            Xt = torch.from_numpy(X).to(device)
            Xt_norm = (Xt - mean_t) / std_t
            preds = np.empty(N, dtype=np.float32)
            bs = 65536
            for i in range(0, N, bs):
                yb = model(Xt_norm[i:i+bs]).cpu().numpy().ravel()
                preds[i:i+bs] = yb

            # 6) Colocar en el tensor de salida
            pred_h = pred[h_idx]
            pred_h[mask] = preds
            pred[h_idx] = pred_h

            print(f"    hora {h_idx:2d} ({int(h_vals[h_idx]):2d}h local): "
                  f"{N:,} píxeles válidos")

    coords = {"hour_of_day": h_vals, "y": y_vals, "x": x_vals}
    pred_da = xr.DataArray(pred, coords=coords,
                           dims=("hour_of_day", "y", "x"), name="UTCI_pred")
    real_da = xr.DataArray(real, coords=coords,
                           dims=("hour_of_day", "y", "x"), name="UTCI_real")
    ds.close()
    return real_da, pred_da

def compute_metrics(real, pred):
    mask = np.isfinite(real) & np.isfinite(pred)
    r = real[mask]
    p = pred[mask]
    err = p - r
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(mask.sum())}


def plot_map(real, pred, season, hour, save_path):
    """Mapa UTCI real vs pred vs diff para una hora específica."""
    r = real.sel(hour_of_day=hour).values
    p = pred.sel(hour_of_day=hour).values
    d = p - r

    vmin = np.nanmin([np.nanmin(r), np.nanmin(p)])
    vmax = np.nanmax([np.nanmax(r), np.nanmax(p)])

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    im0 = ax[0].imshow(r, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    ax[0].set_title(f"UTCI real · {season} · {hour:02d}h")
    plt.colorbar(im0, ax=ax[0], label="°C")

    im1 = ax[1].imshow(p, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    ax[1].set_title("UTCI predicho")
    plt.colorbar(im1, ax=ax[1], label="°C")

    im2 = ax[2].imshow(d, cmap="RdBu_r", vmin=-3, vmax=3)
    ax[2].set_title("Diferencia (pred - real)")
    plt.colorbar(im2, ax=ax[2], label="°C")

    for a in ax:
        a.set_xlabel("x (m)")
        a.set_ylabel("y (m)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def main(season="dry", ckpt_name="pinn_utci_aburra.pt"):
    print(f"\n=== Evaluación · temporada '{season}' ===")
    ckpt_path = MODELS_DIR / ckpt_name
    print(f"Cargando checkpoint: {ckpt_path}")
    model, mean, std = load_model(ckpt_path)

    real, pred = predict_season(model, mean, std, season)

    metrics = compute_metrics(real.values, pred.values)
    print(f"\nMétricas globales ({season}):")
    print(f"  MAE  = {metrics['mae']:.3f} °C")
    print(f"  RMSE = {metrics['rmse']:.3f} °C")
    print(f"  R²   = {metrics['r2']:.4f}")
    print(f"  N    = {metrics['n']:,}")

    # Métricas por hora
    print(f"\nMétricas por hora del día:")
    print(f"  {'hora':>4} | {'MAE':>6} | {'RMSE':>6}")
    for h in real.hour_of_day.values:
        r_h = real.sel(hour_of_day=h).values
        p_h = pred.sel(hour_of_day=h).values
        m = compute_metrics(r_h, p_h)
        print(f"  {int(h):>4} | {m['mae']:>6.3f} | {m['rmse']:>6.3f}")

    # Mapas para horas clave
    for h in [6, 12, 15, 18]:
        save_path = OUTPUT_DIR / f"map_{season}_{h:02d}h.png"
        plot_map(real, pred, season, h, save_path)
        print(f"  Mapa guardado: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="dry",
                        choices=["dry", "wet", "dry2", "wet2"])
    parser.add_argument("--ckpt", default="pinn_utci_aburra.pt")
    args = parser.parse_args()
    main(season=args.season, ckpt_name=args.ckpt)