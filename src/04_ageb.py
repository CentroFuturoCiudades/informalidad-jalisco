"""
04_ageb.py
----------
Aplica la razón informal/formal (de 03_razon.py) a los empleos formales
por AGEB para estimar el número de empleos informales por AGEB y categoría.

La lógica central es:
    informales_AGEB_cat = formales_AGEB_cat × razón(municipio, cat)

donde razón(municipio, cat) viene de 03_razon.py y responde:
"por cada formal en este municipio y categoría, cuántos informales hay".

AGEBs rurales o con cero empleos formales en una categoría producen
cero informales estimados en esa categoría — se incluyen en el output.

Supuestos:
    - La razón informal/formal es homogénea dentro de cada municipio para
      una categoría dada. Todos los AGEBs de un municipio heredan la misma
      razón. La variación espacial intra-municipal la aporta la distribución
      de empleos formales por AGEB.
    - AGEBs con razón no válida (formales ≤ 0 a nivel municipal en 03_razon.py)
      reciben NaN en lugar de un estimado — se marca con una bandera.
    - Se conserva la columna geometry para que el output pueda usarse
      directamente como GeoDataFrame si se carga con geopandas.

Inputs:
    data/processed/razon_informal_formal.csv   (de 03_razon.py)
    data/raw/agebs_empleo_jalisco.gpkg

Outputs:
    data/processed/informales_por_ageb.gpkg
        Una fila por AGEB con columnas:
        - clave_ageb             : identificador único del AGEB
        - clave_entidad          : clave INEGI de entidad
        - clave_municipio        : clave INEGI del municipio
        - clave_localidad        : clave INEGI de localidad
        - ageb                   : clave corta del AGEB
        - nombre_municipio       : nombre del municipio
        - tipo_ageb              : 'urbano' o 'rural'
        - empleos_M              : formales M (input)
        - empleos_S              : formales S (input)
        - empleos_P              : formales P (input)
        - empleos_G              : formales G (input)
        - informales_M           : estimado de informales en M
        - informales_S           : estimado de informales en S
        - informales_P           : estimado de informales en P
        - informales_G           : estimado de informales en G
        - confiable_M/S/P/G      : True si la razón usada es confiable
        - razon_valida_M/S/P/G   : True si la razón es aplicable (formales > 0)
        - geometry               : geometría del AGEB (pass-through)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import geopandas as gpd
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["M", "S", "P", "G"]

# ── Carga ─────────────────────────────────────────────────────────────────────

razon = pd.read_csv(PROCESSED / "razon_informal_formal.csv")
ageb = gpd.read_file(Path("data/raw") / "agebs_empleo_jalisco.gpkg")
ageb["clave_ageb"] = ageb["clave_ageb"].astype(str)

print(f"AGEBs cargados:  {len(ageb):,}")
print(
    f"Razones cargadas: {len(razon):,}  ({razon['municipio'].nunique()} municipios × {razon['clasificacion_empleo'].nunique()} categorías)"
)

# ── Pivotear razones a formato ancho ──────────────────────────────────────────

# Necesitamos una fila por municipio con columnas razon_M, razon_S, razon_P, razon_G
# para poder hacer un merge limpio contra los AGEBs.
razon_wide = razon.pivot_table(
    index=["mun", "municipio"],
    columns="clasificacion_empleo",
    values=["razon_informal_formal", "razon_valida", "confiable"],
).reset_index()

# Aplanar MultiIndex de columnas: (razon_informal_formal, M) → razon_M
razon_wide.columns = [
    col[0] if col[1] == "" else f"{col[0].replace('razon_informal_formal', 'razon')}_{col[1]}"
    for col in razon_wide.columns
]

print(f"\nColumnas de razones (wide): {razon_wide.columns.tolist()}")

# ── Cruzar AGEBs con razones por municipio ────────────────────────────────────

# clave_municipio en AGEBs es la clave INEGI (ej. 39 para Guadalajara)
# mun en razon_wide es la misma clave — el merge es directo.
ageb_merged = ageb.merge(
    razon_wide,
    left_on="clave_municipio",
    right_on="mun",
    how="left",
)

n_sin_razon = ageb_merged["mun"].isna().sum()
if n_sin_razon > 0:
    municipios_faltantes = ageb[ageb_merged["mun"].isna()]["nombre_municipio"].unique()
    print(f"\nAGEBs sin razón disponible: {n_sin_razon} ({municipios_faltantes})")
    print("  → informales quedarán como NaN para esos AGEBs")

# ── Estimar informales por categoría ──────────────────────────────────────────

# informales_cat = formales_cat × razón_cat
# Si razón no válida o no disponible → NaN (no inventamos un número)
for cat in CATEGORIES:
    formales_col = f"empleos_{cat}"
    razon_col = f"razon_{cat}"
    valida_col = f"razon_valida_{cat}"

    # Aplicar razón solo donde es válida; NaN en caso contrario
    ageb_merged[f"informales_{cat}"] = np.where(
        ageb_merged[valida_col].fillna(False),
        (ageb_merged[formales_col] * ageb_merged[razon_col]).round().astype("Int64"),
        pd.NA,
    )

# ── Tabla final ───────────────────────────────────────────────────────────────

cols_output = (
    [
        "clave_ageb",
        "clave_entidad",
        "clave_municipio",
        "clave_localidad",
        "ageb",
        "nombre_municipio",
        "tipo_ageb",
    ]
    + [f"empleos_{cat}" for cat in CATEGORIES]
    + [f"informales_{cat}" for cat in CATEGORIES]
    + [f"confiable_{cat}" for cat in CATEGORIES]
    + [f"razon_valida_{cat}" for cat in CATEGORIES]
    + ["geometry"]
)

out = ageb_merged[cols_output].copy()

out = gpd.GeoDataFrame(out, geometry="geometry", crs=ageb.crs)
out.to_file(PROCESSED / "informales_por_ageb.gpkg", driver="GPKG")
print(f"\nGuardado: informales_por_ageb.gpkg ({len(out):,} AGEBs)")

# ── Resumen ───────────────────────────────────────────────────────────────────

print("\nTotales estimados ZMG:")
for cat in CATEGORIES:
    formales = out[f"empleos_{cat}"].sum()
    informales = out[f"informales_{cat}"].sum(skipna=True)
    n_nulos = out[f"informales_{cat}"].isna().sum()
    print(
        f"  {cat}: {formales:>10,.0f} formales  |  {informales:>10,.0f} informales estimados  |  {n_nulos} AGEBs sin estimado"
    )

print("\nAGEBs con razón no válida o sin cruce:")
for cat in CATEGORIES:
    valida = out[f"razon_valida_{cat}"].astype("boolean")
    n = valida.isna().sum() + (~valida.fillna(True)).sum()
    print(f"  {cat}: {n} AGEBs")

print("\nAGEBs con estimado no confiable (razón válida pero fuente débil):")
for cat in CATEGORIES:
    confiable = out[f"confiable_{cat}"].astype("boolean")
    valida = out[f"razon_valida_{cat}"].astype("boolean")
    n = (~confiable.fillna(False) & valida.fillna(False)).sum()
    print(f"  {cat}: {n} AGEBs")
