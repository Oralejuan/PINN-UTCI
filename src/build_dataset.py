"""Integración de GEE + ERA5-Land y cálculo de UTCI.

Estrategia MVP: agregamos ERA5 por hora-del-día (climatología horaria)
para colapsar la dimensión temporal de ~2500 pasos a 24. Sin esto, el
downscaling a 30 m requiere >30 GiB por variable.

Biblioteca pura: recibe paths/Datasets, devuelve Datasets.
"""

import numpy as np
import xarray as xr
import rioxarray as rxr

from config.settings import PATHS, TARGET_YEAR, SEASONS


# ──────────────────────────────────────────────────────────────
# Carga de GeoTIFFs GEE
# ──────────────────────────────────────────────────────────────
def load_gee_stack(path):
    """Carga un GeoTIFF multi-banda preservando la dimensión 'band'."""
    return rxr.open_rasterio(path)


BANDS_SEASON = ["LST", "NDVI", "FVC", "ALBEDO", "EMIS"]
BANDS_DEM    = ["elevation", "slope", "aspect"]


def harmonize_gee_layers(season="dry"):
    """Carga y reproyecta LST, NDVI, FVC, ALBEDO, EMIS + DEM + landcover."""
    base = PATHS["gee"]

    dem_img = load_gee_stack(base / "aburra_dem.tif")
    lc_img  = load_gee_stack(base / f"aburra_landcover_{TARGET_YEAR}.tif")
    sea_img = load_gee_stack(base / f"aburra_{season}_{TARGET_YEAR}.tif")

    sea_img = sea_img.rio.reproject_match(dem_img)
    lc_img  = lc_img.rio.reproject_match(dem_img)

    def _sel(img, idx, name):
        return img.isel(band=idx).drop_vars("band", errors="ignore").rename(name)

    ds = xr.Dataset({
        "LST":       _sel(sea_img, 0, "LST"),
        "NDVI":      _sel(sea_img, 1, "NDVI"),
        "FVC":       _sel(sea_img, 2, "FVC"),
        "ALBEDO":    _sel(sea_img, 3, "ALBEDO"),
        "EMIS":      _sel(sea_img, 4, "EMIS"),
        "elevation": _sel(dem_img, 0, "elevation"),
        "slope":     _sel(dem_img, 1, "slope"),
        "aspect":    _sel(dem_img, 2, "aspect"),
        "landcover": _sel(lc_img, 0, "landcover").astype("int16"),
    })
    
    # ── Recorte automático: eliminar filas/columnas 100% NaN ──
    valid = np.isfinite(ds["LST"].values)
    col_ok = valid.any(axis=0)
    row_ok = valid.any(axis=1)
    j0, j1 = np.where(col_ok)[0][[0, -1]]
    i0, i1 = np.where(row_ok)[0][[0, -1]]
    ds = ds.isel(x=slice(j0, j1 + 1), y=slice(i0, i1 + 1))
    
    return ds


# ──────────────────────────────────────────────────────────────
# Agregación temporal ERA5 → climatología horaria
# ──────────────────────────────────────────────────────────────
def aggregate_era5_hourly(era5_ds, season):
    """Colapsa ERA5 a 24 pasos (uno por hora del día) para una temporada.

    Soporta rangos cross-year: si doy_start > doy_end, toma días de
    (TARGET_YEAR-1) para la parte final del año y (TARGET_YEAR) para
    la parte inicial.
    """
    from config.settings import TARGET_YEAR

    if season not in SEASONS:
        raise ValueError(f"Temporada '{season}' no está en SEASONS")

    doy_start, doy_end = SEASONS[season]
    doy = era5_ds["time"].dt.dayofyear
    year = era5_ds["time"].dt.year

    if doy_start <= doy_end:
        # Rango contiguo dentro del año target
        mask = (doy >= doy_start) & (doy <= doy_end) & (year == TARGET_YEAR)
    else:
        # Cross-year
        prev = (doy >= doy_start) & (year == TARGET_YEAR - 1)
        curr = (doy <= doy_end) & (year == TARGET_YEAR)
        mask = prev | curr

    ds_season = era5_ds.isel(time=mask)

    if ds_season.sizes["time"] == 0:
        raise ValueError(f"Sin datos ERA5 para temporada '{season}'")

    ds_hourly = ds_season.groupby("time.hour").mean(dim="time", keep_attrs=True)
    ds_hourly = ds_hourly.rename({"hour": "hour_of_day"})
    return ds_hourly

# ──────────────────────────────────────────────────────────────
# Downscaling ERA5 → grilla GEE (ahora con solo 24 timesteps)
# ──────────────────────────────────────────────────────────────
def downscale_era5(era5_ds, target_grid, method="bilinear"):
    """Reproyecta ERA5-Land (lat/lon) a la grilla del DEM (UTM 30 m).

    Ahora el cubo temporal es 24x más chico gracias a aggregate_era5_hourly.
    """
    ds = era5_ds

    rename_map = {}
    for candidate, canonical in [("longitude", "x"), ("lon", "x"),
                                  ("latitude", "y"), ("lat", "y")]:
        if candidate in ds.coords and candidate != canonical:
            rename_map[candidate] = canonical
    if rename_map:
        print(f"[downscale] renombrando: {rename_map}")
        ds = ds.rename(rename_map)

    ds = ds.rio.write_crs("EPSG:4326")
    
    ds_utm = ds.rio.reproject_match(target_grid, resampling=_get_resampling(method))
    return ds_utm


def _get_resampling(method):
    from rasterio.enums import Resampling
    return {
        "bilinear": Resampling.bilinear,
        "nearest":  Resampling.nearest,
        "cubic":    Resampling.cubic,
    }.get(method, Resampling.bilinear)


# ──────────────────────────────────────────────────────────────
# Variables derivadas
# ──────────────────────────────────────────────────────────────
def rh_from_dewpoint(T_air_C, T_dew_C):
    """Humedad relativa (%). Magnus-Tetens."""
    a, b = 17.625, 243.04
    rh = 100.0 * np.exp((a * T_dew_C) / (b + T_dew_C) - (a * T_air_C) / (b + T_air_C))
    return np.clip(rh, 0.0, 100.0)


def estimate_mrt(T_air_C, LST_C, SSRD, STRD, STR,
                 albedo_p=0.37, emis_p=0.97, f_p=0.22, F_sky=0.5):
    sigma = 5.67e-8
    S_abs = albedo_p * SSRD * f_p
    L_down = STRD
    L_up = sigma * (LST_C + 273.15) ** 4
    L_abs = emis_p * (L_down * F_sky + L_up * (1.0 - F_sky))
    
    # Clamp: la raíz cuarta necesita un input estrictamente positivo
    net_rad = np.maximum(S_abs + L_abs, 1e-6)   # ← NUEVA LÍNEA
    
    mrt_K = (net_rad / (emis_p * sigma)) ** 0.25
    return mrt_K - 273.15


def compute_utci(T_air_C, MRT_C, wind_speed_ms, RH):
    """UTCI vectorizado. Devuelve np.ndarray con la forma de las entradas."""
    from pythermalcomfort.models import utci

    tdb = np.asarray(T_air_C, dtype=float)
    tr  = np.asarray(MRT_C, dtype=float)
    v   = np.asarray(wind_speed_ms, dtype=float)
    rh  = np.asarray(RH, dtype=float)

    shape = tdb.shape
    f_tdb, f_tr, f_v, f_rh = (a.ravel() for a in (tdb, tr, v, rh))
    valid = (np.isfinite(f_tdb) & np.isfinite(f_tr)
             & np.isfinite(f_v) & np.isfinite(f_rh))

    out = np.full(f_tdb.shape, np.nan, dtype=float)
    if valid.any():
        res = utci(tdb=f_tdb[valid], tr=f_tr[valid],
                   v=f_v[valid], rh=f_rh[valid], units="SI")
        vals = getattr(res, "utci", res)
        out[valid] = np.asarray(vals)

    return out.reshape(shape)


def wrap_like(arr, template):
    """Envuelve un np.ndarray en xr.DataArray con la malla de `template`."""
    return xr.DataArray(arr, coords=template.coords, dims=template.dims,
                        name=template.name)