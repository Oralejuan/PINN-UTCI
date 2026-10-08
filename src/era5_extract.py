# src/era5_extract.py

import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi
import xarray as xr
import numpy as np

from config.settings import TARGET_YEAR, PATHS, get_aoi


VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "total_precipitation",
]

# CDS permite ~4–10 peticiones concurrentes por usuario. 4 es seguro.
MAX_WORKERS = 4


def _download_chunk(year, month, day_start, day_end, out_path, area):
    if out_path.exists():
        return out_path, "skip"

    days = [f"{d:02d}" for d in range(day_start, day_end + 1)]
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": [f"{h:02d}:00" for h in range(0, 24)],
            "area": area,                    # ← argumento
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(out_path),
    )
    return out_path, "ok"


def download_era5_land(year=TARGET_YEAR, out_dir=None):
    out_dir = out_dir or PATHS["era5"]
    out_dir.mkdir(parents=True, exist_ok=True)

    final_path = out_dir / f"era5_land_aburra_{year}.nc"
    if final_path.exists():
        print(f"✅ Ya existe el archivo final: {final_path}")
        return final_path

    # ← NUEVO: calcular AREA desde el AOI
    aoi_info = get_aoi()
    area = aoi_info["bbox_wgs84"]

    # ── Construir bloques: 2 por mes (días 1–15, 16–fin) ──
    chunks = []
    for month in range(1, 13):
        last_day = calendar.monthrange(year, month)[1]
        mid = 15
        chunks.append((month, 1, mid))
        chunks.append((month, mid + 1, last_day))

    tasks = []
    for month, d0, d1 in chunks:
        p = out_dir / f"era5_land_aburra_{year}_{month:02d}_{d0:02d}-{d1:02d}.nc"
        tasks.append((month, d0, d1, p))

    # ── Descarga en paralelo ──
    print(f"🚀 Lanzando {len(tasks)} descargas con {MAX_WORKERS} workers en paralelo...")
    completed = []
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_download_chunk, year, m, d0, d1, p, area): (m, d0, d1, p)
            for (m, d0, d1, p) in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            m, d0, d1, p = futures[fut]
            try:
                _, status = fut.result()
                completed.append(p)
                print(f"  [{i:2d}/{len(tasks)}] {m:02d}/{d0:02d}-{d1:02d} → {status}")
            except Exception as e:
                failed.append((m, d0, d1, str(e)))
                print(f"  [{i:2d}/{len(tasks)}] {m:02d}/{d0:02d}-{d1:02d} → ERROR: {e}")

    if failed:
        print(f"\n⚠️  {len(failed)} bloque(s) fallaron. Vuelve a ejecutar el comando")
        print("   y los bloques ya descargados se saltarán automáticamente.\n")

    if not completed:
        raise RuntimeError("Ningún bloque se descargó correctamente.")

    # ── Unir en un solo NetCDF ──
    print(f"🔗 Uniendo {len(completed)} archivos en {final_path.name}...")
    ds = xr.open_mfdataset(
        [str(p) for p in completed],
        combine="by_coords",
        parallel=False,
    )
    ds.to_netcdf(final_path)
    print(f"✅ Guardado: {final_path}")

    # Opcional: borrar los archivos temporales (descomenta si quieres)
    # for p in completed:
    #     p.unlink()

    return final_path

def download_era5_extra_months(year, months, out_dir=None):
    """Descarga meses específicos de un año (para complementar el dataset).

    Uso típico: bajar dic 2023 para la temporada DJF del año 2024.

    Parámetros
    ----------
    year : int
        Año de los datos (ej. 2023).
    months : list[int]
        Meses a descargar, 1-12 (ej. [12]).
    """
    out_dir = out_dir or PATHS["era5"]
    out_dir.mkdir(parents=True, exist_ok=True)

    months_str = "-".join(f"{m:02d}" for m in sorted(months))
    final_path = out_dir / f"era5_land_extra_{year}_{months_str}.nc"
    if final_path.exists():
        print(f"✅ Ya existe: {final_path}")
        return final_path

    aoi_info = get_aoi()
    area = aoi_info["bbox_wgs84"]

    # Dividir en quincenas para no exceder el límite de CDS
    chunks = []
    for month in months:
        last_day = calendar.monthrange(year, month)[1]
        mid = 15
        chunks.append((month, 1, mid))
        chunks.append((month, mid + 1, last_day))

    tasks = []
    for month, d0, d1 in chunks:
        p = out_dir / f"era5_extra_{year}_{month:02d}_{d0:02d}-{d1:02d}.nc"
        tasks.append((month, d0, d1, p))

    print(f"🚀 Descargando {len(tasks)} bloques extra ({year} meses {months})...")
    completed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_download_chunk, year, m, d0, d1, p, area): (m, d0, d1, p)
            for (m, d0, d1, p) in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            m, d0, d1, p = futures[fut]
            try:
                _, status = fut.result()
                completed.append(p)
                print(f"  [{i:2d}/{len(tasks)}] {m:02d}/{d0:02d}-{d1:02d} → {status}")
            except Exception as e:
                print(f"  [{i:2d}/{len(tasks)}] {m:02d}/{d0:02d}-{d1:02d} → ERROR: {e}")

    if not completed:
        raise RuntimeError("Ningún bloque extra se descargó.")

    print(f"🔗 Uniendo {len(completed)} bloques en {final_path.name}...")
    ds = xr.open_mfdataset([str(p) for p in completed], combine="by_coords", parallel=False)
    ds.to_netcdf(final_path)
    print(f"✅ Guardado: {final_path}")
    return final_path

# ──────────────────────────────────────────────────────────────
# Post-proceso
# ──────────────────────────────────────────────────────────────
def _accum_to_flux(da):
    """Convierte acumulados diarios a flujo horario W/m².

    ERA5-Land reinicia el acumulado entre UTC 00 y UTC 01 de cada día:
      - UTC 00 → total del día anterior (24h acumuladas)
      - UTC 01 → 1h acumulada del día nuevo
      - UTC 02 → 2h acumuladas
      - ...

    Estrategia:
      1. Posición 0 (inicio del año): arr[0] / 86400 (media horaria de 24h)
      2. Resto: diff normal / 3600
      3. Donde el diff sea negativo (reset detectado) → arr[t] / 3600
      4. Clamp a ≥ 0
    """
    da_t = da.transpose("time", ...)
    arr = da_t.values.astype(np.float32)

    flux = np.empty_like(arr)

    # Posición 0: arr[0] es 24h acumuladas del día anterior → media horaria
    flux[0] = arr[0] / 86400.0

    # Resto: diff temporal
    flux[1:] = (arr[1:] - arr[:-1]) / 3600.0

    # Donde el diff es negativo → hubo reset → usar arr[t] / 3600 (1h acum)
    mask_reset = flux < 0
    flux[mask_reset] = arr[mask_reset] / 3600.0

    # Clamp a ≥ 0 (física)
    flux = np.clip(flux, 0.0, None)

    return xr.DataArray(
        flux.astype(da.dtype),
        coords=da_t.coords, dims=da_t.dims, name=da.name, attrs=da.attrs,
    ).transpose(*da.dims)

def load_era5(*paths):
    """Carga y combina uno o más archivos ERA5-Land.

    Ejemplo:
        load_era5(path_2024)
        load_era5(path_2024, path_dec_2023)
    """
    if len(paths) == 1 and isinstance(paths[0], (list, tuple)):
        paths = tuple(paths[0])

    datasets = []
    for p in paths:
        ds_i = xr.open_dataset(p)
        # Normalizar coords
        ren_coords = {}
        for old, new in [("longitude", "lon"), ("latitude", "lat"),
                         ("valid_time", "time")]:
            if old in ds_i.coords or old in ds_i.dims:
                ren_coords[old] = new
        if ren_coords:
            ds_i = ds_i.rename(ren_coords)
        datasets.append(ds_i)

    if len(datasets) > 1:
        ds = xr.concat(datasets, dim="time").sortby("time")
        # Eliminar duplicados de tiempo si los hay
        _, unique_idx = np.unique(ds["time"].values, return_index=True)
        ds = ds.isel(time=np.sort(unique_idx))
    else:
        ds = datasets[0]

    # Renombrar variables
    ren_vars = {
        "t2m": "T_air_K", "d2m": "T_dew_K",
        "u10": "u10", "v10": "v10",
        "ssrd": "SSRD", "strd": "STRD",
        "ssr": "SSR", "str": "STR",
        "tp": "TP",
    }
    ds = ds.rename({k: v for k, v in ren_vars.items() if k in ds.data_vars})

    # Acumulados horarios -> W/m²
    for v in ["SSRD", "STRD", "SSR", "STR"]:
        if v in ds.data_vars:
            ds[v] = _accum_to_flux(ds[v])

    # Derivadas
    ds["T_air_C"] = ds["T_air_K"] - 273.15
    ds["T_dew_C"] = ds["T_dew_K"] - 273.15
    ds["wind_speed"] = np.sqrt(ds["u10"]**2 + ds["v10"]**2)

    return ds


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = download_era5_land()
    ds = load_era5(p)
    print(ds)
    print("\nVariables:", list(ds.data_vars))