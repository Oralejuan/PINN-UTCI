"""Intento de descarga automática de estaciones SIATA.

El portal SIATA no expone históricos por URL estable; la mayoría de
intentos caerán y deberás descargar los CSV manualmente desde
https://siata.gov.co y colocarlos en data/raw/siata/.
Este módulo sigue funcionando como loader y estandarizador.
"""

import pandas as pd
import requests

from config.settings import TARGET_YEAR, PATHS

SIATA_BASE = "https://siata.gov.co/data/storeAllavas/"

# Estaciones meteorológicas clave del valle (código SIATA)
STATIONS = {
    "MED001": "Medellín Centro",
    "MED002": "Medellín El Poblado",
    "BEL001": "Bello",
    "ENV001": "Envigado",
    "ITA001": "Itagüí",
    "SAB001": "Sabaneta",
    "CAL001": "Caldas",
    "COP001": "Copacabana",
}


def download_station_csv(station_code, year=TARGET_YEAR, out_dir=None):
    """Intenta descargar CSV de una estación. Si falla, retorna None."""
    out_dir = out_dir or PATHS["siata"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{station_code}_{year}.csv"
    if out_path.exists():
        return out_path
    try:
        url = f"{SIATA_BASE}{year}/{station_code}.csv"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return out_path
    except Exception as e:
        print(f"[WARN] {station_code}: {e}. Descarga manual desde https://siata.gov.co")
        return None


def load_station(path):
    """Carga un CSV SIATA y estandariza columnas."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for k, v in cols.items():
        if "temperatura" in k:
            rename[v] = "T_air_C"
        elif "humedad" in k:
            rename[v] = "RH"
        elif "viento" in k and "vel" in k:
            rename[v] = "wind_speed"
        elif "radiacion" in k:
            rename[v] = "R_global"
        elif "fecha" in k:
            rename[v] = "timestamp"
    df = df.rename(columns=rename)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


if __name__ == "__main__":
    ok, fail = 0, 0
    for code in STATIONS:
        p = download_station_csv(code)
        if p is None:
            fail += 1
        else:
            ok += 1
    print(f"\nDescargadas: {ok} | Fallidas: {fail}")
    if fail:
        print(f"Descarga manual requerida. Coloca los CSV en: {PATHS['siata']}")