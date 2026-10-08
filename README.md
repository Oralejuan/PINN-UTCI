# 🌡️ PINN-UTCI Valle de Aburrá

**Physics-Informed Neural Network para la estimación de estrés térmico urbano en el Valle de Aburrá (Colombia)**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 📋 Tabla de Contenidos

1. [Descripción](#-descripción)
2. [Motivación Científica](#-motivación-científica)
3. [Características Principales](#-características-principales)
4. [Arquitectura del Proyecto](#-arquitectura-del-proyecto)
5. [Instalación](#-instalación)
6. [Configuración Inicial](#-configuración-inicial)
7. [Uso](#-uso)
8. [Pipeline Detallado](#-pipeline-detallado)
9. [Metodología](#-metodología)
10. [Resultados](#-resultados)
11. [Limitaciones](#-limitaciones)
12. [Trabajo Futuro](#-trabajo-futuro)
13. [Referencias](#-referencias)
14. [Estructura de Datos](#-estructura-de-datos)
15. [Contribuciones](#-contribuciones)
16. [Licencia](#-licencia)

---

## 📖 Descripción

Este proyecto implementa una **Physics-Informed Neural Network (PINN)** para estimar el **Universal Thermal Climate Index (UTCI)** en el **Valle de Aburrá**, Antioquia, Colombia. Combina:

- **Datos satelitales** (Landsat 8/9, SRTM, Dynamic World)
- **Reanálisis meteorológico** (ERA5-Land)
- **Ecuaciones físicas** (difusión de calor + advección topográfica)

El objetivo es producir mapas de estrés térmico a **30 m de resolución espacial** y **resolución horaria**, utilizables para:

- Investigación en climatología urbana
- Planificación urbana y arborización
- Sistemas de alerta temprana por calor (Early Action Protocols)
- Análisis de vulnerabilidad térmica

### 🎯 Dominio de Estudio

| Parámetro | Valor |
|---|---|
| **Área** | Valle de Aburrá, Antioquia, Colombia |
| **Municipios** | 10 (Barbosa → Caldas) |
| **Área total** | ~1,158 km² |
| **Bounding box** | N=6.51°, S=5.98°, W=−75.72°, E=−75.22° |
| **Altitud** | 1,300 – 3,000 msnm |
| **Clima** | Tropical andino, régimen bimodal |
| **Población** | ~4 millones (10 municipios) |

---

## 🔬 Motivación Científica

### ¿Por qué PINN y no un modelo puramente data-driven?

Las redes neuronales tradicionales (MLP, CNN, RNN) pueden aprender UTCI a partir de datos, pero:

1. **Requieren enormes cantidades de datos** etiquetados.
2. **No respetan leyes físicas** — pueden predecir temperaturas imposibles.
3. **Generalizan mal** fuera del dominio de entrenamiento.
4. **Son cajas negras** sin interpretabilidad física.

Una PINN incorpora las **ecuaciones gobernantes** (en este caso, la ecuación de transporte de calor) como término de regularización en la función de pérdida:

$$
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{datos}}}_{\text{MSE ponderado}} + \lambda_p \underbrace{\mathcal{L}_{\text{física}}}_{\text{residuo PDE}} + \lambda_s \underbrace{\mathcal{L}_{\text{suavizado}}}_{\text{inercia temporal}}
$$

**Beneficios:**
- ✅ Menos datos requeridos
- ✅ Predicciones físicamente consistentes
- ✅ Mejor generalización
- ✅ Regularización natural contra sobreajuste

### ¿Por qué UTCI?

El **Universal Thermal Climate Index (UTCI)** es el índice de estrés térmico más completo para espacios exteriores:

- Combina **temperatura del aire, humedad, viento y radiación** en un solo índice
- Basado en un **modelo termorregulatorio humano** (Fiala)
- Expresado en **°C equivalentes**, interpretable fisiológicamente
- Estándar internacional (ISB Commission 6)

Rangos de estrés:

| UTCI (°C) | Estrés térmico |
|---|---|
| < −13 | Frío extremo |
| −13 a 0 | Frío fuerte |
| 0 a 9 | Frío moderado |
| 9 a 26 | **Sin estrés térmico** |
| 26 a 32 | Estrés por calor moderado |
| 32 a 38 | Estrés por calor fuerte |
| 38 a 46 | Estrés por calor muy fuerte |
| > 46 | Estrés por calor extremo |

---

## ✨ Características Principales

| Característica | Descripción |
|---|---|
| 🛰️ **Datos abiertos** | Landsat, SRTM, ERA5-Land, Dynamic World |
| 🌍 **Reproducible** | AOI cargado desde GeoJSON |
| 📅 **Temporadas cross-year** | DJF (dic año ant. + ene-feb) |
| 🧠 **PINN con física** | Difusión + advección topográfica |
| ⚖️ **Weighted loss** | Balance por Effective Sample Size |
| 🎯 **Split espacial** | Holdout con K-means para validación honesta |
| 📊 **Métricas completas** | MAE, RMSE, R² por hora y temporada |
| 🗺️ **Mapas automáticos** | Real vs predicho vs diferencia |
| 🖥️ **Modular** | CLI por etapas, extensible |

---

## 🏗️ Arquitectura del Proyecto

```
pinn_utci_aburra/
├── config/
│   └── settings.py              # Parámetros globales, rutas, AOI, temporadas
├── data/
│   ├── aoi/                     # Archivo GeoJSON del AOI
│   ├── raw/                     # Datos crudos descargados
│   │   ├── gee/                 # GeoTIFFs de Google Earth Engine
│   │   ├── era5/                # NetCDF de ERA5-Land
│   │   └── siata/               # CSV de estaciones (opcional)
│   ├── interim/                 # Datos intermedios
│   └── processed/               # NetCDF finales con UTCI
├── src/
│   ├── aoi.py                   # Carga y valida AOI
│   ├── gee_extract.py           # Exportaciones GEE
│   ├── era5_extract.py          # Descarga/preprocesa ERA5
│   ├── siata_extract.py         # Carga estaciones SIATA
│   ├── build_dataset.py         # Integra fuentes + calcula UTCI
│   ├── run_pipeline.py          # Orquestador ETL
│   └── pinn/
│       ├── __init__.py
│       ├── dataset.py           # Carga .nc → tensores PyTorch
│       ├── model.py             # MLP con tanh
│       ├── losses.py            # Pérdida física + weighted
│       ├── train.py             # Entrenamiento por etapas
│       └── evaluate.py          # Métricas y mapas
├── models/                      # Checkpoints (.pt) + metrics.json
├── outputs/                     # Mapas PNG, reportes
├── notebooks/                   # Jupyter notebooks exploratorios
├── requirements.txt
└── README.md
```

---

## 🚀 Instalación

### Requisitos del Sistema

- **Python** 3.11 o 3.12 (evitar 3.13+ por incompatibilidad de `earthengine-api`)
- **RAM** 8 GB mínimo (16 GB recomendado)
- **Disco** 20 GB libres (datos + checkpoints)
- **OS** Windows 10/11, Linux, macOS

### Paso 1 — Clonar el repositorio

```bash
git clone <tu-repo-url>
cd pinn_utci_aburra
```

### Paso 2 — Crear entorno virtual

**Windows:**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Paso 3 — Instalar dependencias

```bash
pip install -r requirements.txt
```

**Contenido mínimo de `requirements.txt`:**

```
# Core científico
numpy>=1.24
pandas>=2.0
xarray>=2023.1
rioxarray>=0.15
netCDF4>=1.6
dask>=2023.1

# Geoespacial
geopandas>=0.14
shapely>=2.0
pyogrio>=0.7
rasterio>=1.3

# Descarga de datos
earthengine-api>=0.1.380
cdsapi>=0.6

# Red neuronal
torch>=2.0
scikit-learn>=1.3

# Índices térmicos
pythermalcomfort>=2.9

# Visualización
matplotlib>=3.7

# Utilidades
requests>=2.31
psutil>=5.9
```

### Paso 4 — Crear los `__init__.py`

```bash
# Linux/macOS
touch config/__init__.py src/__init__.py src/pinn/__init__.py

# Windows
New-Item config\__init__.py -ItemType File
New-Item src\__init__.py -ItemType File
New-Item src\pinn\__init__.py -ItemType File
```

---

## ⚙️ Configuración Inicial

### 1. Google Earth Engine

```bash
earthengine authenticate
```

1. Se abre el navegador → inicia sesión con tu cuenta Google
2. Copia el token y pégalo en la terminal
3. **Registra tu proyecto** en https://code.earthengine.google.com/register
4. Añade tu project ID en `config/settings.py`:

```python
EE_PROJECT = "ee-tu-nombre-de-proyecto"
```

### 2. Copernicus CDS (ERA5-Land)

1. Regístrate en https://cds.climate.copernicus.eu/
2. **Acepta los términos** de ERA5-Land en https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land (pestaña Download → Terms of use)
3. Copia tu token desde tu perfil
4. Crea `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: TU-TOKEN-AQUI
```

### 3. Configurar el AOI

Crea tu `data/aoi/valle_aburra.geojson` con:

- **Opción A**: Dibuja en https://geojson.io y guarda el JSON
- **Opción B**: Descarga fronteras administrativas de GADM (https://gadm.org)
- **Opción C**: Exporta desde QGIS con EPSG:4326

Verifica que carga:

```bash
python -m src.aoi
```

**Salida esperada:**
```
AOI: valle_aburra.geojson
  CRS original : EPSG:4326
  Features     : 10
  Área aprox   : 1157.8 km²
  BBox WGS84   : N=6.5119  W=-75.7194  S=5.9782  E=-75.2212
  BBox 9377    : X=[...]  Y=[...] m
```

---

## 🎮 Uso

### Pipeline Completo (primera vez)

```bash
# 1. Lanzar exportaciones en GEE (~30-60 min en servidor)
python -m src.run_pipeline --stage gee

# 2. Monitorear en https://code.earthengine.google.com/tasks
#    Descargar los 6 .tif de Google Drive a data/raw/gee/

# 3. Descargar ERA5-Land (~20-40 min)
python -m src.run_pipeline --stage era5

# 4. Construir el dataset procesado (~20 min)
python -m src.run_pipeline --stage build --seasons dry wet dry2 wet2

# 5. Entrenar la PINN (~10-40 min)
python -u -m src.run_pinn --epochs 500 --n-samples 1000000 --split-mode spatial

# 6. Evaluar y generar mapas
python -m src.pinn.evaluate --season dry
```

### Uso por Etapas

```bash
# Solo verificar el AOI
python -m src.aoi

# Re-descargar ERA5 solo para meses extra
python -c "
from src.era5_extract import download_era5_extra_months
from config.settings import PATHS, TARGET_YEAR
download_era5_extra_months(year=TARGET_YEAR-1, months=[12], out_dir=PATHS['era5'])
"

# Reconstruir una sola temporada
python -m src.run_pipeline --stage build --seasons dry

# Entrenar con split aleatorio (para comparar)
python -u -m src.run_pinn --epochs 500 --n-samples 1000000 --split-mode random

# Evaluar cada temporada
python -m src.pinn.evaluate --season wet
python -m src.pinn.evaluate --season dry2
python -m src.pinn.evaluate --season wet2
```

### Opciones del Entrenamiento

| Argumento | Valores | Descripción |
|---|---|---|
| `--epochs` | int | Épocas totales (default: 500) |
| `--batch-size` | int | Tamaño de batch (default: 16384) |
| `--lr` | float | Learning rate (default: 1e-3) |
| `--n-samples` | int | Puntos objetivo (default: 200,000) |
| `--split-mode` | `random`, `spatial` | Tipo de split (default: random) |

---

## 🔄 Pipeline Detallado

### Etapa 1 — GEE (Google Earth Engine)

**Entrada:** Polígono AOI en `data/aoi/`

**Salida:** 6 archivos GeoTIFF en `data/raw/gee/`

| Archivo | Descripción |
|---|---|
| `aburra_dem.tif` | Elevación + pendiente + aspecto (SRTM) |
| `aburra_landcover_2024.tif` | Cobertura (Dynamic World) |
| `aburra_dry_2024.tif` | LST, NDVI, FVC, albedo, emisividad |
| `aburra_wet_2024.tif` | ... |
| `aburra_dry2_2024.tif` | ... |
| `aburra_wet2_2024.tif` | ... |

**Preprocesamiento aplicado:**
- Máscara de nubes (`QA_PIXEL` bits 1–4)
- Escala oficial de reflectancia C2 L2
- Cálculo de índices: NDVI, FVC, albedo banda ancha (Liang 2001), emisividad (Sobrino)
- Compuesto mediano por temporada

### Etapa 2 — ERA5-Land

**Entrada:** Coordenadas del AOI

**Salida:** `data/raw/era5/era5_land_aburra_2024.nc` + `era5_land_extra_2023_12.nc`

**Variables descargadas:**

| Variable | Descripción | Uso |
|---|---|---|
| `t2m` | Temperatura del aire a 2 m | UTCI |
| `d2m` | Punto de rocío a 2 m | RH |
| `u10`, `v10` | Viento a 10 m | Velocidad viento |
| `ssrd`, `strd` | Radiación solar/térmica descendente | MRT |
| `ssr`, `str` | Radiación neta | Balance |

**Preprocesamiento:**
- Conversión de acumulados diarios a flujo horario (W/m²)
- Detección y corrección del reset a UTC 01
- Clamp ≥ 0 en variables físicamente no negativas

### Etapa 3 — Build Dataset

**Entrada:** GEE + ERA5

**Salida:** `data/processed/aburra_{season}_utci.nc` + `_local.nc`

**Proceso:**
1. Armonización de GEE (reproyección a grilla del DEM)
2. Agregación de ERA5 a climatología horaria (24 pasos)
3. Downscaling ERA5 a 30 m con `rioxarray`
4. Cálculo de RH (Magnus-Tetens)
5. Cálculo de MRT (Thorsson et al. 2007)
6. Cálculo de UTCI (pythermalcomfort)
7. Conversión UTC → hora local (UTC−5)

### Etapa 4 — Entrenamiento PINN

**Entrada:** 4 archivos `.nc` procesados

**Salida:** `models/pinn_utci_aburra.pt` + `models/metrics.json`

**Proceso:**
1. Carga y muestreo uniforme por hora
2. Split train/val/test (random o espacial)
3. Cálculo de pesos por temporada (ESS)
4. Etapa 1: 250 epochs solo datos
5. Etapa 2: 250 epochs datos + física (rampa λ_p)
6. Evaluación final sobre test

### Etapa 5 — Evaluación

**Salida:** `outputs/map_{season}_{hour}h.png` (3 paneles: real, pred, diff)

---

## 🧪 Metodología

### Definición de Temporadas

Basado en el régimen bimodal del Valle de Aburrá (validado en literatura climatológica):

| Temporada | DOY | Meses | Descripción |
|---|---|---|---|
| `dry` | 335–59 | dic (año ant.) + ene-feb | Temporada seca DJF |
| `wet` | 60–151 | mar-may | Primer pico de lluvias (MAM) |
| `dry2` | 152–212 | jun-jul | Veranillo de San Juan |
| `wet2` | 213–334 | ago-nov | Segundo pico de lluvias (SON) |

**Nota:** La temporada `dry` cruza el año siguiendo la definición estándar de DJF en climatología.

### Arquitectura de la Red

| Parámetro | Valor | Justificación |
|---|---|---|
| Tipo | MLP | Simplicidad, suficiencia para MVP |
| Capas ocultas | 4 | Balance capacidad/sobreajuste |
| Neuronas/capa | 128 | Estándar en problemas similares |
| Activación | `tanh` | Derivadas suaves (crítico para PINN) |
| Entrada | 18 features | Coordenadas + variables físicas |
| Salida | 1 (UTCI) | |

### Features de Entrada

| # | Feature | Origen |
|---|---|---|
| 0 | `x` | Coordenada UTM 18N |
| 1 | `y` | Coordenada UTM 18N |
| 2 | `hour_of_day` | 0–23 hora local |
| 3 | `season_id` | 0=dry, 1=wet, 2=dry2, 3=wet2 |
| 4 | `elevation` | SRTM |
| 5 | `slope` | SRTM |
| 6 | `aspect` | SRTM |
| 7 | `landcover` | Dynamic World |
| 8 | `LST` | Landsat |
| 9 | `NDVI` | Landsat |
| 10 | `FVC` | Calculado de NDVI |
| 11 | `ALBEDO` | Landsat (Liang) |
| 12 | `EMIS` | Landsat (Sobrino) |
| 13 | `T_air_C` | ERA5-Land |
| 14 | `RH` | ERA5-Land |
| 15 | `wind_speed` | ERA5-Land |
| 16 | `SSRD` | ERA5-Land |
| 17 | `MRT` | Calculado |

### Función de Pérdida

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{datos}} + \lambda_p \mathcal{L}_{\text{física}} + \lambda_s \mathcal{L}_{\text{suavizado}}
$$

**Componente de datos (weighted):**

$$
\mathcal{L}_{\text{datos}} = \frac{1}{B}\sum_{i=1}^{B} w_i \, (y_i - \hat{y}_i)^2
$$

donde $w_i = \frac{N_{\text{total}}/4}{N_{\text{temporada}(i)}}$ corrige el desbalance por cobertura satelital.

**Componente física (residuo PDE):**

$$
\mathcal{L}_{\text{física}} = \left\langle \left| \frac{\partial T}{\partial t} - \alpha \nabla^2 T - \beta \, \nabla z \cdot \nabla T \right|^2 \right\rangle
$$

**Componente de suavizado temporal:**

$$
\mathcal{L}_{\text{suavizado}} = \left\langle \left| T(t+\Delta t) - T(t) \right|^2 \right\rangle
$$

### Estrategia de Entrenamiento

**Split de datos:** 80% train / 10% val / 10% test

**Dos modos:**
- `random`: split aleatorio sobre puntos
- `spatial`: K-means (K=5) sobre coordenadas, 3 clusters train / 1 val / 1 test

**Etapas:**
1. **Etapa 1** (250 epochs): $\lambda_p = 0$, $\lambda_s = 0$ — aprendizaje data-driven
2. **Etapa 2** (250 epochs): $\lambda_p$ rampa 0 → 0.5, $\lambda_s = 0.05$ — refinamiento físico

---

## 📊 Resultados

### Métricas de Validación (split espacial)

| Métrica | Valor | Interpretación |
|---|---|---|
| **MAE test** | 0.044 °C | Error absoluto medio |
| **RMSE test** | 0.058 °C | Raíz del error cuadrático |
| **R² test** | 0.9997 | Coeficiente de determinación |

**Nota metodológica:** El MAE tan bajo se explica porque el modelo recibe como entradas `T_air_C`, `MRT`, `RH` y `wind_speed`, que son los cuatro inputs de la fórmula de UTCI. Para un análisis de ablación (eliminar estas features), ver [Trabajo Futuro](#-trabajo-futuro).

### Cobertura por Temporada

| Temporada | UTCI NaN | LST NaN | Causa |
|---|---|---|---|
| `dry` | 70.2% | 67.4% | Nubes orográficas |
| `wet` | 91.4% | 91.3% | Nubosidad convectiva |
| `dry2` | 69.8% | 66.0% | Veranillo |
| `wet2` | 80.6% | 79.6% | Segundo pico de lluvia |

**La cobertura útil varía de ~9% (wet) a ~30% (dry).** Esto es un límite físico del sensor Landsat, no del pipeline.

### Mapas Generados

Cada temporada genera 4 PNGs (06h, 12h, 15h, 18h locales) con 3 paneles:

1. **UTCI real** (target)
2. **UTCI predicho** (PINN)
3. **Diferencia** (pred − real)

Los archivos se guardan en `outputs/map_{season}_{hour}h.png`.

---

## ⚠️ Limitaciones

Reconocemos las siguientes limitaciones del MVP actual:

### Del Modelo

1. **Aprendizaje de la fórmula UTCI**: al recibir `T_air_C`, `MRT`, `RH` y `wind_speed` como entradas, la red aprende la función determinista UTCI. El MAE de 0.044 °C refleja esto, no una generalización espacial completa.
2. **Física simplificada**: la advección topográfica `β∇z·∇T` es una aproximación, no la ecuación completa de Navier-Stokes.
3. **MRT aproximada**: asume `F_sky=0.5` constante; en realidad depende de obstrucción urbana (edificios, árboles).

### De los Datos

4. **Nubosidad**: Landsat tiene ~30% de píxeles útiles en `dry` y ~9% en `wet` en el valle. Esto es un límite del sensor.
5. **ERA5-Land a 9 km**: el reanálisis es más grueso que la resolución de la grilla (30 m). El downscaling produce variabilidad espacial aparente.
6. **Un solo año (2024)**: no captura variabilidad interanual completa.
7. **Sin validación contra estaciones in-situ**: pendiente de integrar SIATA.

### Del Análisis

8. **Sin quantificación de incertidumbre**: el modelo actual da predicciones puntuales sin intervalos.
9. **Sin validación temporal**: solo se validó espacialmente. Entrenar con 3 temporadas y testear con la 4ª está pendiente.

---

## 🔬 Trabajo Futuro

### Corto Plazo

- [ ] **Ablación de features**: entrenar sin `T_air_C`, `MRT`, `RH`, `wind_speed` para medir el valor real del modelo
- [ ] **Validación temporal**: holdout por temporada
- [ ] **Integración SIATA**: descarga manual de estaciones meteorológicas
- [ ] **Interfaz Streamlit**: control desde navegador

### Mediano Plazo

- [ ] **MC Dropout**: intervalos de predicción (uncertainty quantification)
- [ ] **Múltiples años** (2023, 2025): generalización temporal
- [ ] **Sky View Factor desde DEM**: mejora de MRT
- [ ] **Análisis de calibración**: reliability diagrams, Brier score
- [ ] **Relleno físico de zonas con NaN**: cobertura 100%

### Largo Plazo (Early Action Protocol)

- [ ] **Modelo operacional**: asimilación de datos en tiempo real
- [ ] **Umbrales con margen de seguridad**: alertas calibradas
- [ ] **Marco ético-operacional**: decisiones ante incertidumbre
- [ ] **Integración con SIATA**: sistema municipal de alertas
- [ ] **Validación con eventos históricos extremos**

---

## 📚 Referencias

### Metodología PINN

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686–707.

### UTCI

- Bröde, P., et al. (2012). *Deriving the operational procedure for the Universal Thermal Climate Index (UTCI)*. International Journal of Biometeorology, 56(3), 481–494.

### MRT

- Thorsson, S., et al. (2007). *Different methods for estimating the mean radiant temperature in an outdoor urban setting*. International Journal of Climatology, 27(14), 1983–1993.

### Índices de vegetación y albedo

- Carlson, T. N., & Ripley, D. A. (1997). *On the relation between NDVI, fractional vegetation cover, and leaf area index*. Remote Sensing of Environment, 62(3), 241–252.
- Liang, S. (2001). *Narrowband to broadband conversions of land surface albedo I: Algorithms*. Remote Sensing of Environment, 76(2), 213–238.
- Sobrino, J. A., et al. (2004). *Land surface temperature retrieval from LANDSAT TM 5*. Remote Sensing of Environment, 90(4), 434–440.

### Climatología Valle de Aburrá

- Bedoya-Soto, J. M., et al. (2019). *Seasonal shift of the diurnal cycle of rainfall over Medellin's valley, Central Andes of Colombia*. Frontiers in Earth Science.
- SIATA (Sistema de Alerta Temprana de Medellín y el Valle de Aburrá). Reportes climáticos anuales.

### Weighted loss

- Wang, Z., et al. (2021). *A unified framework for dataset bias*. NeurIPS.

---

## 📂 Estructura de Datos

### Datos crudos (`data/raw/`)

**GEE** (`data/raw/gee/`):
- `aburra_dem.tif` — 3 bandas: elevation, slope, aspect
- `aburra_landcover_2024.tif` — 1 banda
- `aburra_{season}_2024.tif` — 5 bandas: LST_C, NDVI, FVC, ALBEDO, EMIS

**ERA5** (`data/raw/era5/`):
- `era5_land_aburra_2024.nc` — año completo
- `era5_land_extra_2023_12.nc` — dic 2023 para DJF

### Datos procesados (`data/processed/`)

- `aburra_{season}_utci.nc` — UTC hora
- `aburra_{season}_utci_local.nc` — hora local (UTC−5)

**Variables por archivo:**
- `UTCI` — target
- `MRT`, `LST`, `T_air_C`, `RH`, `wind_speed`, `SSRD` — features
- `elevation`, `slope`, `aspect`, `landcover` — estáticas
- `NDVI`, `FVC`, `ALBEDO`, `EMIS` — superficie

### Modelos (`models/`)

- `pinn_utci_aburra.pt` — checkpoint final
- `pinn_utci_aburra_stage1.pt` — checkpoint tras Etapa 1
- `metrics.json` — métricas de test

---

## 🤝 Contribuciones

Este proyecto está en desarrollo activo. Si quieres contribuir:

1. Fork el repositorio
2. Crea una rama de feature (`git checkout -b feature/nueva-mejora`)
3. Commit con mensajes claros
4. Push a la rama
5. Abre un Pull Request

### Áreas de interés

- Mejoras en la física (MRT con Sky View Factor)
- Integración con SIATA / IDEAM
- Interfaz de usuario (Streamlit, Dash)
- Optimización de cómputo (GPU, paralelización)
- Validación con eventos extremos

---

## 📄 Licencia

MIT License — ver [LICENSE](LICENSE) para detalles.

---

## 👤 Autor

**Juan Manuel Luna** (2026)

Proyecto desarrollado como parte de investigación en climatología urbana y machine learning físico para el Valle de Aburrá.

---

## 🙏 Agradecimientos

- **Google Earth Engine** por el acceso gratuito a Landsat y SRTM
- **Copernicus Climate Data Store** por ERA5-Land
- **SIATA** por los datos de estaciones meteorológicas
- **Comunidad PINN** (Raissi et al., DeepXDE, etc.)

---

## 📞 Contacto

Para preguntas, sugerencias o colaboraciones:

- 📧 Email: [tu-email]
- 🐛 Issues: [enlace a GitHub Issues]
- 💬 Discussions: [enlace a GitHub Discussions]

---

**Última actualización:** Octubre 2026
**Versión:** 1.0.0 (MVP)