"""Carga el AOI desde un archivo vectorial (GeoJSON, SHP, KML, GPKG).

Centraliza:
- Leer el archivo.
- Reproyectar a WGS84 (EPSG:4326, grados) para GEE y ERA5.
- Calcular bbox WGS84 (para CDS) y bbox proyectado (para reportes).
- Exponer geometría shapely y GeoJSON para GEE.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import shapely
from shapely.geometry import mapping
from shapely.ops import unary_union


@lru_cache(maxsize=4)
def load_aoi(path):
    """
    Devuelve dict con:
      - 'shapely'      : geometría EPSG:4326 (grados) para GEE
      - 'gdf'          : GeoDataFrame original
      - 'bbox_wgs84'   : [N, W, S, E] en GRADOS (para CDS ERA5)
      - 'bbox_proj'    : [minx, miny, maxx, maxy] en EPSG:9377 (metros, para reportes)
      - 'geojson'      : dict GeoJSON listo para GEE
      - 'crs_original' : CRS del archivo de entrada
      - 'area_km2'     : área en km²
      - 'n_features'   : número de features del archivo
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el AOI: {path}")

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"AOI vacío: {path}")

    crs_original = str(gdf.crs)

    # ─── 1. Reproyectar a WGS84 (grados) para GEE + ERA5 ───
    gdf_wgs = gdf.to_crs("EPSG:4326")
    geom_wgs = unary_union(gdf_wgs.geometry.values)
    if not geom_wgs.is_valid:
        geom_wgs = shapely.make_valid(geom_wgs)

    minx, miny, maxx, maxy = geom_wgs.bounds
    bbox_wgs84 = [maxy, minx, miny, maxx]   # [N, W, S, E] — formato CDS

    # ─── 2. Reproyectar a UTM 18N (metros) para reportes/área ───
    gdf_proj = gdf.to_crs("EPSG:32618")
    geom_proj = unary_union(gdf_proj.geometry.values)
    bbox_proj = list(geom_proj.bounds)      # [minx, miny, maxx, maxy]
    area_km2 = geom_proj.area / 1e6

    # ─── 3. GeoJSON para GEE ───
    geojson = mapping(geom_wgs)

    return {
        "shapely": geom_wgs,
        "gdf": gdf,
        "bbox_wgs84": bbox_wgs84,
        "bbox_proj": bbox_proj,
        "geojson": geojson,
        "crs_original": crs_original,
        "area_km2": area_km2,
        "n_features": len(gdf),
    }


def summarize(path) -> str:
    info = load_aoi(path)
    n, w, s, e = info["bbox_wgs84"]
    minx, miny, maxx, maxy = info["bbox_proj"]
    return (
        f"AOI: {Path(path).name}\n"
        f"  CRS original : {info['crs_original']}\n"
        f"  Features     : {info['n_features']}\n"
        f"  Área aprox   : {info['area_km2']:.1f} km²\n"
        f"  BBox WGS84   : N={n:.4f}  W={w:.4f}  S={s:.4f}  E={e:.4f}\n"
        f"  BBox 9377    : X=[{minx:.0f}, {maxx:.0f}]  Y=[{miny:.0f}, {maxy:.0f}] m"
    )


if __name__ == "__main__":
    import sys
    from config.settings import AOI_PATH
    p = sys.argv[1] if len(sys.argv) > 1 else AOI_PATH
    print(summarize(p))