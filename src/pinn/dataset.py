"""Carga los .nc procesados y construye tensores para la PINN.

Novedades:
- Split train/val/test (80/10/10) reproducible.
- Muestreo uniforme por hora del día (cada hora contribuye igual).
- Media/std calculados solo sobre train.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from config.settings import PATHS


FEATURE_COLS = [
    "elevation", "slope", "aspect",
    "NDVI", "FVC", "ALBEDO", "EMIS", "LST", "landcover",
    "T_air_C", "RH", "wind_speed", "SSRD", "MRT",
]
TARGET_COL = "UTCI"

SEASON_IDS = {"dry": 0, "wet": 1, "dry2": 2, "wet2": 3}

_FULL_CACHE = {}


def _load_full(seasons, n_samples, seed):
    """Carga X crudo + y + índices de split. Cachea por proceso."""
    key = (tuple(seasons), n_samples, seed)
    if key in _FULL_CACHE:
        return _FULL_CACHE[key]

    import xarray as xr

    rng = np.random.default_rng(seed)
    all_X = []
    all_y = []

    for si, season in enumerate(seasons):
        # Preferir el archivo local; caer al UTC si no existe
        nc_path = PATHS["processed"] / f"aburra_{season}_utci_local.nc"
        if not nc_path.exists():
            nc_path = PATHS["processed"] / f"aburra_{season}_utci.nc"
        if not nc_path.exists():
            print(f"[WARN] Falta {nc_path}, se omite")
            continue

        ds = xr.open_dataset(nc_path)

        n_h = ds.sizes.get("hour_of_day", 1)
        n_x = ds.sizes["x"]
        n_y = ds.sizes["y"]
        n_cells = n_x * n_y

        # ── Muestreo uniforme por hora del día ──
        k_per_season = n_samples // len(seasons)
        k_per_hour = max(1, k_per_season // n_h)

        idx_parts = []
        for h in range(n_h):
            n_per_h = min(k_per_hour, n_cells)
            idx_h = rng.choice(n_cells, size=n_per_h, replace=False)
            idx_parts.append(h * n_cells + idx_h)
        idx = np.concatenate(idx_parts)
        k = len(idx)

        # Decodificar índices lineales → (h, x, y)
        h_idx = idx // n_cells
        xy_idx = idx % n_cells
        x_idx = xy_idx // n_y
        y_idx = xy_idx % n_y

        # DataArrays para indexación nombrada (por dim, no por posición)
        h_da = xr.DataArray(h_idx, dims="points")
        x_da = xr.DataArray(x_idx, dims="points")
        y_da = xr.DataArray(y_idx, dims="points")

        def extract(name, ds=ds, k=k, h_da=h_da, x_da=x_da, y_da=y_da):
            """Extrae una variable para los k puntos, manejando dims 2D y 3D."""
            if name not in ds:
                return np.zeros(k, dtype=np.float32)
            da = ds[name]
            sel = {}
            if "hour_of_day" in da.dims:
                sel["hour_of_day"] = h_da
            if "x" in da.dims:
                sel["x"] = x_da
            if "y" in da.dims:
                sel["y"] = y_da
            return da.isel(sel).values.astype(np.float32)

        # Coordenadas físicas
        xs = ds["x"].values[x_idx].astype(np.float32)
        ys = ds["y"].values[y_idx].astype(np.float32)
        if "hour_of_day" in ds.dims:
            hours = ds["hour_of_day"].values[h_idx].astype(np.float32)
        else:
            hours = np.zeros(k, dtype=np.float32)

        # Features
        feats_list = []
        missing_cols = []
        for c in FEATURE_COLS:
            if c in ds:
                feats_list.append(extract(c))
            else:
                feats_list.append(np.zeros(k, dtype=np.float32))
                missing_cols.append(c)
        if missing_cols:
            print(f"[WARN] {season}: faltan {missing_cols}, se rellenan con 0")
        feats = np.stack(feats_list, axis=1)

        target = extract(TARGET_COL)
        season_col = np.full(k, si, dtype=np.float32)

        # Filtro NaNs
        mask = (np.isfinite(feats).all(1)
                & np.isfinite(target)
                & np.isfinite(xs)
                & np.isfinite(ys))

        X = np.concatenate([
            xs[mask, None], ys[mask, None], hours[mask, None],
            season_col[mask, None], feats[mask],
        ], axis=1).astype(np.float32)
        y = target[mask][:, None].astype(np.float32)

        all_X.append(X)
        all_y.append(y)
        print(f"[dataset] {season}: {len(y):,} válidos de {k:,} muestreados")
        ds.close()

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Split 80/10/10 reproducible
        # ── Reproducibilidad ──
    n = len(y)

    # ── Split aleatorio (referencia) ──
    rng_split = np.random.default_rng(seed + 999)
    perm = rng_split.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    split_random = {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }

    # ── Split espacial (K-means sobre x, y) ──
    from sklearn.cluster import KMeans
    n_clusters = 5
    xy = X[:, :2]  # x, y en UTM
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_ids = km.fit_predict(xy)

    rng_folds = np.random.default_rng(seed + 7)
    fold_order = rng_folds.permutation(n_clusters)
    test_cluster = int(fold_order[0])
    val_cluster = int(fold_order[1])
    train_clusters = fold_order[2:]

    split_spatial = {
        "train": np.where(np.isin(cluster_ids, train_clusters))[0],
        "val":   np.where(cluster_ids == val_cluster)[0],
        "test":  np.where(cluster_ids == test_cluster)[0],
    }

    # Reporte de splits
    print(f"[split] random  → train={len(split_random['train']):>7,} | "
          f"val={len(split_random['val']):>6,} | test={len(split_random['test']):>6,}")
    print(f"[split] spatial → train={len(split_spatial['train']):>7,} | "
          f"val={len(split_spatial['val']):>6,} | test={len(split_spatial['test']):>6,}")
    print(f"[split] spatial → clusters: train={list(train_clusters)} | "
          f"val={val_cluster} | test={test_cluster}")
    for c in range(n_clusters):
        n_c = int((cluster_ids == c).sum())
        role = "test" if c == test_cluster else ("val" if c == val_cluster else "train")
        print(f"[split]   cluster {c}: N={n_c:>7,}  → {role}")

    # ── Pesos por muestra (para weighted loss) ──
    # Balanceo por ESS: sample con temporada rara pesa más.
    season_ids = X[:, 3].astype(np.int32)          # columna 3 = season_id
    season_counts = np.bincount(season_ids, minlength=4).astype(np.float32)
    season_counts = np.maximum(season_counts, 1.0)  # evitar /0

    n_total = len(y)
    n_seasons = 4
    # Peso = (total/n_seasons) / N_season_del_sample
    raw_w = (n_total / n_seasons) / season_counts[season_ids]
    # Normalizar a media 1
    raw_w = raw_w / raw_w.mean()
    sample_weights = raw_w.astype(np.float32)

    # Reporte
    for i, sid in enumerate(sorted(set(season_ids.tolist()))):
        cnt = int(season_counts[sid])
        w = float(raw_w[season_ids == sid][0])
        print(f"[weights] season {sid}: N={cnt:>7,}  w={w:.3f}")

    # Media/std SOLO del train (del split aleatorio, es el más grande)
    X_train = X[split_random["train"]]
    mean = X_train.mean(0, keepdims=True).astype(np.float32)
    std = X_train.std(0, keepdims=True).astype(np.float32) + 1e-8

    result = {
        "X": X,
        "y": y,
        "split_random": split_random,
        "split_spatial": split_spatial,
        "mean": mean,
        "std": std,
        "weights": sample_weights,
        "cluster_ids": cluster_ids,
    }
    _FULL_CACHE[key] = result
    return result


class UTCIDataset(Dataset):
    """Dataset con split reproducible (train/val/test/all)."""

    def __init__(self, seasons=("dry", "wet", "dry2", "wet2"),
                 n_samples=200_000, seed=42, split="train",
                 split_mode="random"):
        """
        Parámetros
        ----------
        split : {'train', 'val', 'test', 'all'}
        split_mode : {'random', 'spatial'}
            - random : split aleatorio (referencia)
            - spatial: K-means espacial (holdout geográfico)
        """
        assert split in {"train", "val", "test", "all"}
        assert split_mode in {"random", "spatial"}
        self.split = split
        self.split_mode = split_mode

        full = _load_full(seasons, n_samples, seed)
        self.mean = full["mean"]
        self.std = full["std"]

        if split_mode == "random":
            idx_map = full["split_random"]
        else:
            idx_map = full["split_spatial"]

        if split == "all":
            idx = np.arange(len(full["y"]))
        else:
            idx = idx_map[split]

        X_raw = full["X"][idx]
        y_raw = full["y"][idx]
        weights = full["weights"][idx]

        X_norm = (X_raw - self.mean) / self.std

        self.X = torch.from_numpy(X_norm.astype(np.float32))
        self.y = torch.from_numpy(y_raw.astype(np.float32))
        self.X_raw = torch.from_numpy(X_raw.astype(np.float32))
        self.weights = torch.from_numpy(weights.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.weights[i]


def load_datasets(n_samples=200_000, seed=42, split_mode="random"):
    """Devuelve (ds_train, ds_val, ds_test) para el modo de split indicado."""
    ds_train = UTCIDataset(n_samples=n_samples, seed=seed,
                            split="train", split_mode=split_mode)
    ds_val = UTCIDataset(n_samples=n_samples, seed=seed,
                          split="val", split_mode=split_mode)
    ds_test = UTCIDataset(n_samples=n_samples, seed=seed,
                           split="test", split_mode=split_mode)
    return ds_train, ds_val, ds_test