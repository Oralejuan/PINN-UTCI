"""
Orquestador del pipeline de datos para la PINN de UTCI del Valle de Aburrá.

Etapas:
    1. gee      → lanza tareas asíncronas de exportación en GEE
    2. era5     → descarga ERA5-Land
    3. siata    → descarga datos SIATA (fallback: manual)
    4. build    → integra y calcula UTCI → data/processed/*.nc

Uso:
    python -m src.run_pipeline --stage all
    python -m src.run_pipeline --stage era5
    python -m src.run_pipeline --stage build --seasons dry wet
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import xarray as xr

# ──────────────────────────────────────────────────────────────
# Garantizar la raíz del proyecto en sys.path
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PATHS, SEASONS, TARGET_YEAR, get_aoi # noqa: E402

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("pipeline")


# ──────────────────────────────────────────────────────────────
# ETAPA 1 — GEE
# ──────────────────────────────────────────────────────────────
def stage_gee(seasons: list[str]) -> None:
    import ee
    from config.settings import PATHS, SEASONS, TARGET_YEAR, get_aoi
    from src import gee_extract as ge

    LOG.info("Etapa 1/4 · GEE — lanzando exportaciones para %s", seasons)
    ge.ensure_init()
    aoi_info = get_aoi()
    aoi = ee.Geometry(aoi_info["geojson"])

    ge.export_dem(aoi)
    ge.export_dynamic_world(aoi)
    for s in seasons:
        if s not in SEASONS:
            LOG.warning("Temporada '%s' no está en SEASONS. Se omite.", s)
            continue
        ge.export_stack(aoi, s, SEASONS[s])
        time.sleep(2)

    LOG.info(
        "Tareas lanzadas. Revísalas en https://code.earthengine.google.com/tasks"
    )
    LOG.info("Descarga los GeoTIFFs resultantes a: %s", PATHS["gee"])


# ──────────────────────────────────────────────────────────────
# ETAPA 2 — ERA5
# ──────────────────────────────────────────────────────────────
def stage_era5() -> None:
    from src import era5_extract as ee

    LOG.info("Etapa 2/4 · ERA5-Land — descargando %s", TARGET_YEAR)
    nc_path = ee.download_era5_land(year=TARGET_YEAR, out_dir=PATHS["era5"])

    # Extra: dic del año anterior (para temporada DJF cross-year)
    LOG.info("Etapa 2/4 · ERA5-Land — descargando dic %s (cross-year)", TARGET_YEAR - 1)
    dec_path = ee.download_era5_extra_months(
        year=TARGET_YEAR - 1, months=[12], out_dir=PATHS["era5"]
    )

    ds = ee.load_era5(nc_path, dec_path)
    LOG.info("ERA5 OK — dims=%s, variables=%s",
             dict(ds.sizes), list(ds.data_vars))


# ──────────────────────────────────────────────────────────────
# ETAPA 3 — SIATA
# ──────────────────────────────────────────────────────────────
def stage_siata() -> None:
    from src import siata_extract as se

    LOG.info("Etapa 3/4 · SIATA — descargando estaciones")
    ok, fail = 0, 0
    for code in se.STATIONS:
        p = se.download_station_csv(code, year=TARGET_YEAR, out_dir=PATHS["siata"])
        if p is None:
            fail += 1
        else:
            ok += 1
    LOG.info("SIATA — %d descargadas, %d fallidas", ok, fail)
    if fail:
        LOG.warning("Descarga manual requerida. Coloca los CSV en: %s", PATHS["siata"])


# ──────────────────────────────────────────────────────────────
# ETAPA 4 — build_dataset
# ──────────────────────────────────────────────────────────────
def stage_build(seasons: list[str]) -> None:
    """Integra GEE + ERA5 y calcula UTCI. Guarda un .nc por temporada."""
    from src import build_dataset as bd
    from src import era5_extract as ee

    LOG.info("Etapa 4/4 · build_dataset — integrando y calculando UTCI")

    era5_path = PATHS["era5"] / f"era5_land_aburra_{TARGET_YEAR}.nc"
    if not era5_path.exists():
        raise FileNotFoundError(f"Falta ERA5: {era5_path}")

        # Cargar ambos: año completo + dic del año anterior (para DJF)
    dec_prev_path = PATHS["era5"] / f"era5_land_extra_{TARGET_YEAR - 1}_12.nc"
    if dec_prev_path.exists():
        era5 = ee.load_era5(era5_path, dec_prev_path)
        LOG.info("ERA5 cargado con dic %d (cross-year)", TARGET_YEAR - 1)
    else:
        era5 = ee.load_era5(era5_path)
        LOG.warning("Sin dic %d — temporada DJF quedará incompleta", TARGET_YEAR - 1)
    
    LOG.info("ERA5 cargado: %d timesteps", era5.sizes["time"])

    # Variables que se conservan en el dataset final (target + features para la PINN)
    KEEP_VARS = [
        # Entradas espaciales / superficie
        "elevation", "slope", "aspect", "landcover",
        "LST", "NDVI", "FVC", "ALBEDO", "EMIS",
        # Meteorología
        "T_air_C", "RH", "wind_speed",
        # Radiación
        "SSRD", "MRT",
        # Target
        "UTCI",
    ]

    for s in seasons:
        out_path = PATHS["processed"] / f"aburra_{s}_utci.nc"
        if out_path.exists():
            LOG.info("  [skip] %s ya existe (%.0f MB)",
                     out_path.name, out_path.stat().st_size / 1e6)
            continue

        LOG.info("  Procesando temporada '%s'...", s)
        try:
            # 1. Capas GEE (estáticas + LST de temporada)
            ds_gee = bd.harmonize_gee_layers(s)

            # 2. ERA5 → climatología horaria (24 pasos)
            era5_hourly = bd.aggregate_era5_hourly(era5, s)
            LOG.info("    ERA5 climatología horaria: %d horas",
                     era5_hourly.sizes["hour_of_day"])

            # 3. Downscale a 30 m
            ref = ds_gee["elevation"]
            ds_era = bd.downscale_era5(era5_hourly, ref)
            LOG.info("    ERA5 downscaled: %s", dict(ds_era.sizes))

            # 4. Variables derivadas
            ds_era["RH"] = bd.rh_from_dewpoint(
                ds_era["T_air_C"], ds_era["T_dew_C"]
            )
            ds_era["MRT"] = bd.estimate_mrt(
                ds_era["T_air_C"], ds_gee["LST"],
                ds_era["SSRD"], ds_era["STRD"], ds_era["STR"],
            )
            utci_vals = bd.compute_utci(
                ds_era["T_air_C"], ds_era["MRT"],
                ds_era["wind_speed"], ds_era["RH"],
            )
            ds_era["UTCI"] = bd.wrap_like(utci_vals, ds_era["T_air_C"])

            # 5. Empaquetar
            out = xr.merge([ds_gee, ds_era], compat="override")

            # 6. Filtrar solo las variables que necesita la PINN
            keep = [v for v in KEEP_VARS if v in out.data_vars]
            missing = [v for v in KEEP_VARS if v not in out.data_vars]
            if missing:
                LOG.warning("    Variables no encontradas: %s", missing)
            out = out[keep]

            out.attrs["season"] = s
            out.attrs["description"] = "PINN-UTCI inputs + target, Valle de Aburrá"
            out.attrs["year"] = TARGET_YEAR

            # 7. Guardar con compresión
            encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
            out.to_netcdf(out_path, encoding=encoding)

            size_mb = out_path.stat().st_size / 1e6
            LOG.info("  ✔ %s  dims=%s  (%.0f MB)",
                     out_path.name, dict(out.sizes), size_mb)

        except Exception as e:
            LOG.error("  ✖ Falló '%s': %s", s, e)
            LOG.debug(traceback.format_exc())

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de datos PINN-UTCI Valle de Aburrá"
    )
    parser.add_argument(
        "--stage", default="all",
        choices=["all", "gee", "era5", "siata", "build"],
        help="Etapa a ejecutar (default: all)",
    )
    parser.add_argument(
        "--seasons", nargs="+", default=list(SEASONS.keys()),
        help="Temporadas a procesar (default: todas las de settings)",
    )
    args = parser.parse_args()

    LOG.info("═" * 60)
    LOG.info("Pipeline PINN-UTCI · stage=%s · seasons=%s", args.stage, args.seasons)
    LOG.info("Project root: %s", PROJECT_ROOT)
    LOG.info("═" * 60)

    t0 = time.time()
    try:
        if args.stage == "all":
            stage_gee(args.seasons)
            stage_era5()
            stage_siata()
            stage_build(args.seasons)
        elif args.stage == "gee":
            stage_gee(args.seasons)
        elif args.stage == "era5":
            stage_era5()
        elif args.stage == "siata":
            stage_siata()
        elif args.stage == "build":
            stage_build(args.seasons)
    except Exception:
        LOG.exception("Pipeline falló")
        sys.exit(1)

    LOG.info("═" * 60)
    LOG.info("Pipeline terminado en %.1f s", time.time() - t0)
    LOG.info("═" * 60)


if __name__ == "__main__":
    main()