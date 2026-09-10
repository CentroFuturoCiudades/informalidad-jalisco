"""
02_ilmm.py
----------
Procesa el archivo de microdatos del ILMM (Indicadores Laborales para los
Municipios de México) para producir una tabla con la tasa de informalidad
oficial por municipio de Jalisco, junto con su coeficiente de variación
y una bandera de confiabilidad.

El ILMM publica 5 estadísticos por municipio (columna 'est'):
    1 — Estimación puntual (la tasa)
    2 — Error estándar
    3 — Límite inferior del intervalo de confianza al 90%
    4 — Límite superior del intervalo de confianza al 90%
    5 — Coeficiente de variación (CV)

Las tasas del ILMM están expresadas como porcentajes (0-100) y representan:
    pea        → % de la población de 15+ años que es económicamente activa
    ocupados   → % de la PEA que está ocupada
    informales → % de los ocupados que son informales  ← la que usamos

Una estimación es confiable si su CV ≤ 15% (estándar del INEGI para
decidir si una estimación es publicable).

El ILMM usa estimación en áreas pequeñas (Small Area Estimation) combinando
la ENOE con registros del IMSS y el Censo 2020, lo que lo hace más robusto
que una estimación directa de la ENOE para municipios con poca muestra.

Consideraciones y limitaciones de la fuente:
- Temporalidad: Se selecciona el corte de 2023-1T por ser el dato oficial más
  reciente disponible para reflejar la coyuntura del mercado laboral. No se
  promedia con 2022-1T para evitar mezclar temporalidades distintas.
- Estacionalidad: El ILMM solo se publica para el primer trimestre (1T). La tasa
  de informalidad obtenida representa una fotografía del 1T y asume estabilidad
  intertrimestral al carecer de mediciones para 2T, 3T y 4T.
- Variabilidad del CV: La bandera de confiabilidad evalúa la precisión estadística
  específicamente en el 1T-2023; el CV podría fluctuar en otros periodos no observados.

Inputs:
    data/raw/conjunto_de_datos_ilmm_2023_1t.csv

Outputs:
    data/processed/ilmm_jalisco.csv
        Una fila por municipio de Jalisco con columnas:
        - ent                  : clave INEGI de entidad (14 = Jalisco)
        - mun                  : clave INEGI del municipio
        - tasa_pea             : % de población 15+ que es PEA
        - tasa_ocupados        : % de la PEA que está ocupada
        - tasa_informales_ilmm : % de ocupados que son informales (ancla del modelo)
        - cv_informales        : coeficiente de variación de tasa_informales_ilmm
        - confiable_ilmm       : True si cv_informales ≤ 15% (estándar INEGI)
"""

import pandas as pd
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Carga ─────────────────────────────────────────────────────────────────────

# El archivo usa encoding latin1 por caracteres especiales del español
ilmm = pd.read_csv(RAW / "conjunto_de_datos_ilmm_2023_1t.csv", encoding="latin1")

print(f"Registros cargados: {len(ilmm):,}")
print(f"Columnas: {ilmm.columns.tolist()}")
print(f"Estadísticos disponibles (est): {sorted(ilmm['est'].unique())}")

# ── Filtrar Jalisco ───────────────────────────────────────────────────────────

# Clave INEGI de Jalisco es 14
# mun=0 es el total estatal — lo excluimos porque necesitamos municipios
ilmm_jalisco = ilmm[(ilmm["ent"] == 14) & (ilmm["mun"] != 0)].copy()

print(f"\nMunicipios de Jalisco en ILMM: {ilmm_jalisco['mun'].nunique()}")

# ── Extraer estimación puntual (est=1) ────────────────────────────────────────

# La estimación puntual es la tasa de informalidad oficial del municipio.
# Es el ancla que se usa en 03_razon.py para calcular informales absolutos:
#   informales_municipio = ocupados_municipio × (tasa_informales_ilmm / 100)
ilmm_puntual = (
    ilmm_jalisco[ilmm_jalisco["est"] == 1]
    .rename(
        columns={
            "pea": "tasa_pea",  # % PEA sobre población 15+
            "ocupados": "tasa_ocupados",  # % ocupados sobre PEA
            "informales": "tasa_informales_ilmm",  # % informales sobre ocupados
        }
    )
    .drop(columns="est")
    .reset_index(drop=True)
)

# ── Extraer coeficiente de variación (est=5) ──────────────────────────────────

# El CV mide la incertidumbre relativa de la estimación.
# CV = (error estándar / estimación puntual) × 100
# CV ≤ 15%  → estimación confiable (estándar INEGI)
# CV > 15%  → usar con cautela, mayor incertidumbre estadística
ilmm_cv = ilmm_jalisco[ilmm_jalisco["est"] == 5][["ent", "mun", "informales"]].rename(
    columns={"informales": "cv_informales"}
)

# Casteo defensivo a numérico: el INEGI a veces usa caracteres especiales
# (ej. '*', '-') cuando la muestra es insuficiente o los datos son suprimidos.
# 'coerce' convierte esos caracteres a NaN.
ilmm_cv["cv_informales"] = pd.to_numeric(ilmm_cv["cv_informales"], errors="coerce")

# ── Unir estimación puntual y CV ──────────────────────────────────────────────

ilmm_out = ilmm_puntual.merge(ilmm_cv, on=["ent", "mun"])

# Bandera de confiabilidad: True si CV ≤ 15%.
# Si el CV es NaN (dato suprimido por INEGI) la comparación evalúa a False de forma segura.
ilmm_out["confiable_ilmm"] = ilmm_out["cv_informales"] <= 15

# ── Guardar ───────────────────────────────────────────────────────────────────

ilmm_out.to_csv(PROCESSED / "ilmm_jalisco.csv", index=False)
print(f"\nGuardado: ilmm_jalisco.csv ({len(ilmm_out)} filas)")

# ── Resumen de confiabilidad ───────────────────────────────────────────────────

n_confiable = ilmm_out["confiable_ilmm"].sum()
n_total = len(ilmm_out)
print(f"\nMunicipios confiables (CV ≤ 15%): {n_confiable} de {n_total} ({n_confiable/n_total:.0%})")
print(
    f"Municipios no confiables o suprimidos: {n_total - n_confiable} ({(n_total - n_confiable)/n_total:.0%})"
)

print(f"\nTasa de informalidad en Jalisco:")
print(
    f"  Mínima:  {ilmm_out['tasa_informales_ilmm'].min():.1f}%  (mun {ilmm_out.loc[ilmm_out['tasa_informales_ilmm'].idxmin(), 'mun']})"
)
print(
    f"  Máxima:  {ilmm_out['tasa_informales_ilmm'].max():.1f}%  (mun {ilmm_out.loc[ilmm_out['tasa_informales_ilmm'].idxmax(), 'mun']})"
)
print(f"  Mediana: {ilmm_out['tasa_informales_ilmm'].median():.1f}%")

print("\nMunicipios ZMG:")
muns_zmg = {
    39: "guadalajara",
    44: "ixtlahuacan_membrillos",
    70: "el_salto",
    97: "tlajomulco",
    98: "tlaquepaque",
    101: "tonala",
    120: "zapopan",
    124: "zapotlanejo",
}
zmg = ilmm_out[ilmm_out["mun"].isin(muns_zmg.keys())].copy()
zmg["nombre"] = zmg["mun"].map(muns_zmg)
print(
    zmg[["nombre", "tasa_informales_ilmm", "cv_informales", "confiable_ilmm"]].to_string(
        index=False
    )
)
