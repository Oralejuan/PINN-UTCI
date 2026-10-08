from pathlib import Path

# ─── Raíz del proyecto ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Rutas absolutas ───
PATHS = {
    "raw":       PROJECT_ROOT / "data" / "raw",
    "interim":   PROJECT_ROOT / "data" / "interim",
    "processed": PROJECT_ROOT / "data" / "processed",
    "aoi":       PROJECT_ROOT / "data" / "aoi",
}
for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)

PATHS["gee"]   = PATHS["raw"] / "gee"
PATHS["era5"]  = PATHS["raw"] / "era5"
PATHS["siata"] = PATHS["raw"] / "siata"
for p in (PATHS["gee"], PATHS["era5"], PATHS["siata"]):
    p.mkdir(parents=True, exist_ok=True)

# ─── AOI ───
# Ruta al archivo vectorial (GeoJSON/SHP/KML/GPKG).
# Edita este valor si cambias de zona.
AOI_PATH = PATHS["aoi"] / "valle_aburra.geojson"
# ─── Google Earth Engine ───
EE_PROJECT = "ee-juanmalun"   # ← el ID alfanumérico que funcionó

# ─── Carga perezosa del AOI (no se ejecuta al importar) ───
def get_aoi():
    """Carga el AOI. Devuelve dict con shapely, bbox_wgs84, geojson, etc.
    Ver src/aoi.py para el contenido del dict.
    """
    from src.aoi import load_aoi
    return load_aoi(AOI_PATH)


# ─── Configuración temporal ───
TARGET_YEAR = 2024

SEASONS = {
    "dry":  (335, 59),
    "wet":  (60, 151),
    "dry2": (152, 212),
    "wet2": (213, 334),
}

# ─── Rejilla maestra ───
GRID_RES_M = 30
CRS = "EPSG:32618"  