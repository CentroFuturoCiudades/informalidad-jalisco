"""
03_razon.py
-----------
Combina los outputs de 01_enoe.py y 02_ilmm.py para construir la razón
informal/formal por municipio y categoría de empleo (M/S/P/G).

La razón responde: por cada trabajador formal en este municipio y categoría,
¿cuántos informales hay? Por ejemplo, una razón de 0.8 en M en Guadalajara
significa que por cada 10 formales en manufactura hay 8 informales.

Esta razón es el insumo central del paso siguiente (04_ageb.py), donde se
aplica a los empleos formales por AGEB para estimar los informales:

    informales_AGEB_cat = formales_AGEB_cat × razón(municipio, cat)

Por qué no usamos tasa_ENOE × tasa_ILMM directamente:
    La multiplicación de tasas funcionaría si tuviéramos ocupados totales
    por AGEB como punto de partida:
        informales_AGEB = ocupados_AGEB × tasa_ILMM × pct_cat_ENOE
    Pero la fuente de AGEB disponible solo reporta empleos formales, no
    ocupados totales. Aplicar la tasa ILMM (definida sobre ocupados totales)
    directamente sobre formales subestimaría el resultado porque el
    denominador es incorrecto. La razón informal/formal resuelve esto al
    relacionar directamente informales con formales, que es exactamente
    lo que tenemos en los datos de AGEB.

Método en tres pasos:
    1. Anclar el total de informales municipales al ILMM:
           informales_municipio = ocupados_municipio × (tasa_ILMM / 100)
       donde ocupados_municipio viene de la ENOE con survey_weight,
       promediado entre trimestres para mantener escala consistente.

    2. Desagregar por categoría usando la distribución de la ENOE:
           informales_municipio_cat = informales_municipio × pct_cat

    3. Calcular formales y razón:
           formales_municipio_cat = ocupados_municipio_cat - informales_municipio_cat
           razón = informales_municipio_cat / formales_municipio_cat

Supuestos:
    - Los ocupados totales por municipio provienen de la ENOE (promedio
      trimestral con survey_weight). Son el denominador para convertir
      la tasa ILMM en número absoluto de informales.
    - La distribución por categoría de la informalidad (pct_cat) se asume
      homogénea dentro del municipio — todos los AGEBs de un municipio
      heredan la misma razón de su municipio y categoría.
    - El mapeo entre nombre de municipio (ENOE) y clave INEGI (ILMM) se
      extrae directamente del parquet para evitar hardcodeo frágil.

Inputs:
    data/raw/enoe_clasificacion_empleo.parquet          (para extraer mapeo mun→nombre)
    data/processed/ocupados_por_municipio_categoria.csv (de 01_enoe.py)
    data/processed/distribucion_por_categoria.csv       (de 01_enoe.py)
    data/processed/ilmm_jalisco.csv                     (de 02_ilmm.py)

Outputs:
    data/processed/razon_informal_formal.csv
        Una fila por municipio × categoría con columnas:
        - mun                        : clave INEGI del municipio
        - municipio                  : nombre del municipio
        - clasificacion_empleo       : categoría M/S/P/G
        - ocupados                   : ocupados totales (ENOE, promedio trimestral)
        - informales_ilmm            : informales anclados al ILMM
        - formales                   : ocupados - informales_ilmm
        - razon_informal_formal      : informales_ilmm / formales
        - razon_valida               : False si formales ≤ 0 (razón no aplicable)
        - tasa_informales_ilmm       : tasa oficial del ILMM (referencia)
        - cv_informales              : coeficiente de variación del ILMM
        - confiable_ilmm             : True si cv_informales ≤ 15%
        - confiable_enoe             : True si n_muestra ≥ 30 en esa celda
        - confiable                  : True si ambas fuentes son confiables
"""

import pandas as pd
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Mapeo mun → municipio ─────────────────────────────────────────────────────

# Se extrae directamente del parquet para evitar hardcodeo.
# 'otro' se excluye porque agrupa múltiples municipios sin clave única.
df_ref = pd.read_parquet(RAW / "enoe_clasificacion_empleo.parquet", columns=["mun", "municipio"])

mapeo_mun_nombre = (
    df_ref[df_ref["municipio"] != "otro"][["mun", "municipio"]].drop_duplicates().sort_values("mun")
)

print("Mapeo municipio extraído del parquet:")
print(mapeo_mun_nombre.to_string(index=False))

# ── Carga de procesados ───────────────────────────────────────────────────────

ocupados = pd.read_csv(PROCESSED / "ocupados_por_municipio_categoria.csv")
dist = pd.read_csv(PROCESSED / "distribucion_por_categoria.csv")
ilmm = pd.read_csv(PROCESSED / "ilmm_jalisco.csv")

# ── Ocupados totales por municipio ────────────────────────────────────────────

# La tasa ILMM se aplica sobre el total de ocupados del municipio (no por categoría).
# Necesitamos el total municipal para calcular informales absolutos
# antes de desagregar por categoría.
# n_celdas_no_confiables cuenta cuántas celdas municipio×categoría tienen
# muestra insuficiente (n_muestra < 30) — se usa en el resumen final.
ocupados_municipio = ocupados.groupby("municipio", as_index=False).agg(
    ocupados_municipio=("ocupados", "sum"),
    n_celdas_no_confiables=("confiable", lambda x: (~x).sum()),
)

# ── Cruzar ENOE con ILMM ─────────────────────────────────────────────────────

# Paso 1: agregar clave INEGI a la tabla de ocupados municipales
ocupados_municipio = ocupados_municipio.merge(
    mapeo_mun_nombre, on="municipio", how="inner"  # solo municipios identificados — excluye 'otro'
)

# Paso 2: unir con ILMM por clave INEGI
base = ocupados_municipio.merge(
    ilmm[["mun", "tasa_informales_ilmm", "cv_informales", "confiable_ilmm"]], on="mun", how="inner"
)

print(f"\nMunicipios con cruce ENOE-ILMM exitoso: {len(base)}")

# ── Informales municipales anclados al ILMM ───────────────────────────────────

# Convertir tasa porcentual a proporción y multiplicar por ocupados totales.
# Este es el ancla confiable: el total de informales del municipio
# según el ILMM, expresado en número absoluto.
base["informales_municipio_ilmm"] = base["ocupados_municipio"] * base["tasa_informales_ilmm"] / 100

# ── Desagregar por categoría (distribución ENOE) ─────────────────────────────

# pct_informales_en_categoria dice qué fracción del total municipal
# de informales trabaja en cada categoría M/S/P/G.
base_cat = dist.merge(
    base[
        [
            "municipio",
            "mun",
            "ocupados_municipio",
            "informales_municipio_ilmm",
            "tasa_informales_ilmm",
            "cv_informales",
            "confiable_ilmm",
        ]
    ],
    on="municipio",
    how="inner",
)

# Informales por municipio y categoría: aplicar distribución al total ILMM
base_cat["informales_ilmm"] = (
    base_cat["informales_municipio_ilmm"] * base_cat["pct_informales_en_categoria"] / 100
)

# ── Ocupados por categoría para calcular formales ─────────────────────────────

# Unir ocupados por municipio×categoría de la ENOE
base_cat = base_cat.merge(
    ocupados[["municipio", "clasificacion_empleo", "ocupados", "confiable"]].rename(
        columns={"confiable": "confiable_enoe"}
    ),
    on=["municipio", "clasificacion_empleo"],
    how="left",
)

# Formales = ocupados totales de la categoría - informales anclados al ILMM
base_cat["formales"] = base_cat["ocupados"] - base_cat["informales_ilmm"]

# ── Razón informal/formal ─────────────────────────────────────────────────────

# Por cada formal en este municipio y categoría, cuántos informales hay.
# Se usa en 04_ageb.py:
#   informales_AGEB_cat = formales_AGEB_cat × razón(municipio, cat)
base_cat["razon_informal_formal"] = (base_cat["informales_ilmm"] / base_cat["formales"]).round(4)

# Marcar celdas donde formales ≤ 0 — razón no aplicable.
# Ocurre si el ILMM estima más informales que ocupados totales en una categoría,
# lo que puede suceder por inconsistencias de escala entre fuentes.
base_cat["razon_valida"] = base_cat["formales"] > 0

# ── Confiabilidad combinada ───────────────────────────────────────────────────

# Una celda es confiable solo si ambas fuentes lo son:
# - confiable_enoe: n_muestra ≥ 30 en esa celda municipio×categoría (ENOE)
# - confiable_ilmm: CV ≤ 15% para ese municipio (ILMM)
base_cat["confiable"] = base_cat["confiable_enoe"] & base_cat["confiable_ilmm"]

# ── Tabla final ───────────────────────────────────────────────────────────────

razon_out = (
    base_cat[
        [
            "mun",
            "municipio",
            "clasificacion_empleo",
            "ocupados",
            "informales_ilmm",
            "formales",
            "razon_informal_formal",
            "razon_valida",
            "tasa_informales_ilmm",
            "cv_informales",
            "confiable_ilmm",
            "confiable_enoe",
            "confiable",
        ]
    ]
    .sort_values(["municipio", "clasificacion_empleo"])
    .reset_index(drop=True)
)

razon_out.to_csv(PROCESSED / "razon_informal_formal.csv", index=False)
print(f"\nGuardado: razon_informal_formal.csv ({len(razon_out)} filas)")

# ── Resumen ───────────────────────────────────────────────────────────────────

print("\nRazón informal/formal por municipio y categoría:")
print(
    razon_out.pivot(
        index="municipio", columns="clasificacion_empleo", values="razon_informal_formal"
    )
    .round(2)
    .to_string()
)

print(f"\nConfiabilidad de celdas:")
print(f"  Confiables (ENOE + ILMM): {razon_out['confiable'].sum()} de {len(razon_out)}")
print(f"  Razón inválida (formales ≤ 0): {(~razon_out['razon_valida']).sum()}")

# Municipios con al menos una celda de muestra insuficiente en la ENOE
print("\nMunicipios con celdas de muestra insuficiente (n_muestra < 30):")
resumen_no_confiables = ocupados_municipio[ocupados_municipio["n_celdas_no_confiables"] > 0][
    ["municipio", "n_celdas_no_confiables"]
]
if len(resumen_no_confiables) > 0:
    print(resumen_no_confiables.to_string(index=False))
else:
    print("  Ninguna")

# Celdas con razón no válida — requieren atención antes de usar en 04_ageb.py
if (~razon_out["razon_valida"]).any():
    print("\nCeldas con formales ≤ 0 (razón no aplicable en 04_ageb.py):")
    print(
        razon_out[~razon_out["razon_valida"]][
            ["municipio", "clasificacion_empleo", "ocupados", "informales_ilmm", "formales"]
        ].to_string(index=False)
    )
