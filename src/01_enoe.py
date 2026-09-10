"""
01_enoe.py
----------
Procesa la ENOE armonizada `enoe_clasificacion_empleo.parquet`, creada por
Sebastián Gutiérrez Bernal, para producir dos salidas:

1. distribucion_por_categoria.csv
   Distribución porcentual de trabajadores informales por municipio y
   categoría de empleo (M/S/P/G). Responde: "de todos los informales
   de este municipio, qué fracción trabaja en cada categoría". Se usa
   en 03_razon.py para desagregar el total municipal del ILMM en
   categorías.

   Categorías:
     M — Manufactura, construcción, mantenimiento y actividades afines
     S — Ventas y servicios al público
     P — Profesional y directivo
     G — General, administrativo y clerical

2. ocupados_por_municipio_categoria.csv
   Promedio trimestral de ocupados expandidos (con survey_weight) por
   municipio y categoría. Se usa en 03_razon.py como denominador para
   calcular los formales municipales una vez que el ILMM provee los
   informales.

Supuestos:
- Los 8 trimestres (2022t1-2023t4) se colapsan juntos para maximizar
  la muestra y obtener distribuciones más estables.
- Los totales expandidos (ocupados, informales) se dividen entre el
  número de trimestres para expresarlos como promedio trimestral y
  mantener la escala compatible con el ILMM. Nota: la ENOE es un panel
  rotativo con permanencia de 5 trimestres por vivienda; dividir entre
  n_periodos es una aproximación.
- Se excluyen registros con clasificacion_empleo nula (trabajadores sin
  categoría asignable, equivalente al antiguo 'no_especificado') y
  municipio 'otro' (agrupa residuales de Jalisco fuera de la ZMG).
- Se usa survey_weight para todas las estimaciones poblacionales,
  convirtiendo registros muestrales en personas reales.
- La columna informal_weighted (informal × survey_weight) se
  pre-calcula antes del groupby para evitar desalineación de índices
  dentro de lambdas con agg(), que es frágil en pandas moderno.
- Una celda municipio×categoría es confiable si tiene al menos 30
  registros en la muestra (estándar para encuestas de hogares).

Inputs:
    data/raw/enoe_clasificacion_empleo.parquet

Outputs:
    data/processed/distribucion_por_categoria.csv
    data/processed/ocupados_por_municipio_categoria.csv
"""

import pandas as pd
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Carga ─────────────────────────────────────────────────────────────────────

df = pd.read_parquet(RAW / "enoe_clasificacion_empleo.parquet")

# n_periodos se usa después para promediar los acumulados trimestrales
n_periodos = df["period"].nunique()

print(f"Registros cargados:  {len(df):,}")
print(f"Períodos ({n_periodos}):     {sorted(df['period'].unique())}")
print(f"Entidades:           {df['ent'].unique().tolist()}")

# ── Limpieza ──────────────────────────────────────────────────────────────────

# Excluir registros no útiles:
# - clasificacion_empleo nula: trabajadores cuyo SCIAN no mapeó a ninguna
#   categoría M/S/P/G (equivalente al antiguo 'no_especificado'). Incluirlos
#   contaminaría las distribuciones porcentuales.
# - municipio 'otro': agrupa municipios residuales de Jalisco fuera de la ZMG;
#   no tiene correspondencia con claves municipales del ILMM y no puede cruzarse.
mask_excluir = df["clasificacion_empleo"].isna() | (df["municipio"] == "otro")
df_clean = df[~mask_excluir].copy()

print(f"\nRegistros después de limpieza: {len(df_clean):,}")
print(f"Excluidos: {mask_excluir.sum():,} ({mask_excluir.mean():.1%})")
print(f"  - clasificacion_empleo nula: {df['clasificacion_empleo'].isna().sum():,}")
print(f"  - municipio 'otro':          {(df['municipio'] == 'otro').sum():,}")

# Pre-calcular la expansión de informalidad antes del groupby.
# Evita referenciar df_clean.loc[x.index] dentro de una lambda en agg(),
# práctica frágil en pandas moderno que puede desalinear índices o producir NaN.
df_clean["informal_weighted"] = df_clean["informal"] * df_clean["survey_weight"]

# ── Estimación por municipio y categoría ──────────────────────────────────────

# Al colapsar 8 trimestres, la suma directa de pesos acumula ~8× la población
# real (porque las mismas personas aparecen en múltiples trimestres del panel
# rotativo). Guardamos los acumulados y luego dividimos entre n_periodos para
# obtener el promedio trimestral, que es la escala compatible con el ILMM.
estimacion = (
    df_clean.groupby(["municipio", "clasificacion_empleo"])
    .agg(
        # n_muestra: registros en la muestra (sin ponderar).
        # Determina la confiabilidad estadística de la celda.
        n_muestra=("survey_weight", "count"),
        # Acumulados brutos (suma sobre todos los trimestres)
        ocupados_acumulados=("survey_weight", "sum"),
        informales_acumulados=("informal_weighted", "sum"),
    )
    .assign(
        # Promedio trimestral: divide entre n_periodos para que la magnitud
        # sea comparable con cifras anualizadas / trimestrales del ILMM.
        ocupados=lambda x: x["ocupados_acumulados"] / n_periodos,
        informales=lambda x: x["informales_acumulados"] / n_periodos,
        # Tasa de informalidad directa de la ENOE: útil para validación cruzada,
        # pero NO es la ancla del modelo (eso lo provee el ILMM en 03_razon.py).
        tasa_informalidad=lambda x: (x["informales"] / x["ocupados"] * 100).round(1),
        # Celda confiable si tiene al menos 30 observaciones muestrales
        confiable=lambda x: x["n_muestra"] >= 30,
    )
    .reset_index()
)

# ── Salida 1: ocupados por municipio y categoría ──────────────────────────────

# En 03_razon.py esta tabla sirve para calcular:
#   formales = ocupados - informales_anclados_al_ILMM
# Las columnas ocupados e informales están en promedio trimestral,
# escala consistente con el ILMM.
ocupados_out = estimacion[
    [
        "municipio",
        "clasificacion_empleo",
        "n_muestra",
        "ocupados",
        "informales",
        "tasa_informalidad",
        "confiable",
    ]
]

ocupados_out.to_csv(PROCESSED / "ocupados_por_municipio_categoria.csv", index=False)
print(f"\nGuardado: ocupados_por_municipio_categoria.csv ({len(ocupados_out)} filas)")

# ── Salida 2: distribución por categoría de informales por municipio ──────────

# Para cada municipio: qué fracción de sus informales trabaja en cada categoría.
# Ejemplo: en Guadalajara, 35% de los informales están en M.
# Esta distribución se aplicará al total municipal del ILMM para desagregar
# en categorías M/S/P/G.
#
# Nota: al ser un ratio (informales_cat / informales_municipio), el factor
# de escala (÷ n_periodos) se cancela; la distribución porcentual es idéntica
# a la que se obtendría con los acumulados brutos.
total_informales_municipio = estimacion.groupby("municipio")["informales"].transform("sum")

dist_categoria = estimacion[["municipio", "clasificacion_empleo", "informales", "confiable"]].copy()

# pct_raw: porcentaje exacto (sin redondear), usado para la validación.
# pct_informales_en_categoria: versión redondeada para exportación y lectura humana.
dist_categoria["pct_raw"] = dist_categoria["informales"] / total_informales_municipio * 100
dist_categoria["pct_informales_en_categoria"] = dist_categoria["pct_raw"].round(1)

# Verificar que los porcentajes exactos suman 100 por municipio.
# Se usa pct_raw (sin redondear) con tolerancia numérica para evitar falsos
# positivos por acumulación de errores de punto flotante al redondear.
check = dist_categoria.groupby("municipio")["pct_raw"].sum()
assert (
    abs(check - 100) < 1e-5
).all(), f"Los porcentajes no suman 100% en: {check[abs(check - 100) >= 1e-5].index.tolist()}"

dist_categoria_out = dist_categoria[
    ["municipio", "clasificacion_empleo", "informales", "pct_informales_en_categoria", "confiable"]
]
dist_categoria_out.to_csv(PROCESSED / "distribucion_por_categoria.csv", index=False)
print(f"Guardado: distribucion_por_categoria.csv ({len(dist_categoria_out)} filas)")

# ── Resumen de confiabilidad ───────────────────────────────────────────────────

print("\nCeldas no confiables (n_muestra < 30):")
no_confiables = estimacion[~estimacion["confiable"]][
    ["municipio", "clasificacion_empleo", "n_muestra"]
]
print(no_confiables.to_string(index=False) if len(no_confiables) > 0 else "  Ninguna")

print("\nDistribución por categoría por municipio (%):")
print(
    dist_categoria_out.pivot(
        index="municipio", columns="clasificacion_empleo", values="pct_informales_en_categoria"
    ).to_string()
)
