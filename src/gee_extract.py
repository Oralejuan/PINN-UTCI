"""Extracción de capas satelitales desde Google Earth Engine.

Todos los GeoTIFFs se exportan a Google Drive (carpeta PINN_ABURRA);
debes descargarlos manualmente a data/raw/gee/ antes de correr build_dataset.
"""

import time
import ee

from config.settings import TARGET_YEAR, SEASONS, PATHS, GRID_RES_M, CRS

def ensure_init(project=None):
    """Inicializa EE usando EE_PROJECT si está definido."""
    import ee
    from config.settings import EE_PROJECT
    project = project or EE_PROJECT
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)

        


# ──────────────────────────────────────────────────────────────
# Utilidades de procesamiento Landsat C2 L2
# ──────────────────────────────────────────────────────────────
def mask_landsat_c2_l2(img):
    """Enmascara nubes, sombras, cirros y dilate-cloud."""
    qa = img.select("QA_PIXEL")
    cloud  = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)
    cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    dilate = qa.bitwiseAnd(1 << 1).eq(0)
    return img.updateMask(cloud.And(shadow).And(cirrus).And(dilate))


def scale_sr(img):
    """Reflectancia de superficie C2 L2 (rango -0.2 a 1.6)."""
    sr = img.select(["SR_B.*"]).multiply(2.75e-5).add(-0.2)
    return img.addBands(sr, None, True)


def add_indices(img):
    """NDVI, FVC, albedo (Liang 2001) y emisividad (Sobrino)."""
    ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    fvc = ndvi.subtract(0.05).divide(0.81).clamp(0, 1).pow(2).rename("FVC")
    albedo = (
        img.select("SR_B2").multiply(0.356)
        .add(img.select("SR_B4").multiply(0.130))
        .add(img.select("SR_B5").multiply(0.373))
        .add(img.select("SR_B6").multiply(0.085))
        .add(img.select("SR_B7").multiply(0.072))
        .subtract(0.0018)
    ).rename("ALBEDO")
    emis = ee.Image(0).expression(
        "ndvi < 0.2 ? 0.97 : (ndvi > 0.5 ? 0.99 : 0.971 + 0.018*fvc)",
        {"ndvi": ndvi, "fvc": fvc},
    ).rename("EMIS")
    return img.addBands([ndvi, fvc, albedo, emis])


def add_lst_celsius(img):
    """ST_B10 -> °C (escala oficial USGS C2 L2)."""
    lst = (
        img.select("ST_B10")
        .multiply(0.00341802).add(149.0)
        .subtract(273.15).rename("LST_C")
    )
    return img.addBands(lst)


# ──────────────────────────────────────────────────────────────
# Compuestos
# ──────────────────────────────────────────────────────────────
def seasonal_composite(aoi, season_name, doy_range):
    """Compuesto mediano de Landsat 8/9 para una temporada.

    Soporta rangos que cruzan el año: si doy_start > doy_end, combina
    imágenes de doy_start..366 del año TARGET_YEAR-1 con 1..doy_end
    del año TARGET_YEAR.
    """
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    base = l8.merge(l9)

    doy_start, doy_end = doy_range

    if doy_start <= doy_end:
        # Rango contiguo dentro del año target
        col = (base
               .filterBounds(aoi)
               .filter(ee.Filter.calendarRange(TARGET_YEAR, TARGET_YEAR, "year"))
               .filter(ee.Filter.dayOfYear(doy_start, doy_end)))
    else:
        # Cross-year: parte en año TARGET_YEAR-1, parte en año TARGET_YEAR
        prev = (base
                .filterBounds(aoi)
                .filter(ee.Filter.calendarRange(TARGET_YEAR-1, TARGET_YEAR-1, "year"))
                .filter(ee.Filter.dayOfYear(doy_start, 366)))
        curr = (base
                .filterBounds(aoi)
                .filter(ee.Filter.calendarRange(TARGET_YEAR, TARGET_YEAR, "year"))
                .filter(ee.Filter.dayOfYear(1, doy_end)))
        col = prev.merge(curr)

    col = (col
           .filter(ee.Filter.lt("CLOUD_COVER", 60))
           .map(mask_landsat_c2_l2)
           .map(scale_sr)
           .map(add_indices)
           .map(add_lst_celsius))

    return col.median().clip(aoi).set("season", season_name)


# ──────────────────────────────────────────────────────────────
# Exportaciones
# ──────────────────────────────────────────────────────────────
def export_stack(aoi, season_name, doy_range, drive_folder="PINN_ABURRA"):
    """Exporta stack multi-banda: LST_C, NDVI, FVC, ALBEDO, EMIS."""
    img = seasonal_composite(aoi, season_name, doy_range)
    bands = ["LST_C", "NDVI", "FVC", "ALBEDO", "EMIS"]  # ← orden fijo
    task = ee.batch.Export.image.toDrive(
        image=img.select(bands).toFloat(),
        description=f"aburra_{season_name}_{TARGET_YEAR}",
        folder=drive_folder,
        region=aoi,
        scale=GRID_RES_M,
        crs=CRS,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def export_dem(aoi, drive_folder="PINN_ABURRA"):
    """Exporta DEM SRTM + pendiente + aspecto."""
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi).rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem).rename("aspect")
    stack = dem.addBands([slope, aspect]).toFloat()
    task = ee.batch.Export.image.toDrive(
        image=stack,
        description="aburra_dem",
        folder=drive_folder,
        region=aoi,
        scale=GRID_RES_M,
        crs=CRS,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def export_dynamic_world(aoi, year=TARGET_YEAR, drive_folder="PINN_ABURRA"):
    """Exporta cobertura Dynamic World (moda anual)."""
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(aoi)
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .select("label").mode().clip(aoi).rename("landcover")
    )
    task = ee.batch.Export.image.toDrive(
        image=dw.toByte(),
        description=f"aburra_landcover_{year}",
        folder=drive_folder,
        region=aoi,
        scale=10,
        crs=CRS,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config.settings import get_aoi
    ensure_init()
    info = get_aoi()
    aoi = ee.Geometry(info["geojson"])   # ← GeoJSON directo
    export_dem(aoi)
    export_dynamic_world(aoi)
    for s_name, doy in SEASONS.items():
        export_stack(aoi, s_name, doy)
        time.sleep(2)
    print("\nTareas lanzadas. Revísalas en:")
    print("  https://code.earthengine.google.com/tasks")
    print(f"Luego descarga los GeoTIFFs a: {PATHS['gee']}")