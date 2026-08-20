# Databricks notebook source
# MAGIC %md
# MAGIC # Huella geométrica del archivo de piezas
# MAGIC
# MAGIC Recorre el catálogo de STL y extrae, por pieza, un vector de descriptores
# MAGIC que permite (a) agrupar el archivo en familias, (b) decidir qué variante
# MAGIC del modelo aplica, y (c) obtener los parámetros de colada por diferencia
# MAGIC entre la versión con sistema y la versión suelta.
# MAGIC
# MAGIC **No sustituye a la simulación.** Es un censo: barato, tolerante a fallos,
# MAGIC y ejecutable sobre las 1500 piezas en una tarde.
# MAGIC
# MAGIC ## Tres niveles
# MAGIC | Nivel | Fuente | Cobertura |
# MAGIC |---|---|---|
# MAGIC | 1 — geométrico | STL de pieza suelta | todas |
# MAGIC | 2 — colada | (pieza+colada) − pieza | la mayoría |
# MAGIC | 3 — multi-cavidad | racimo | algunas |

# COMMAND ----------

# MAGIC %run ./_core_llenado

# COMMAND ----------

import re
import datetime
import numpy as np
import pandas as pd
import trimesh
from shapely.geometry import box
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StringType, StructType, StructField, DoubleType, IntegerType, BooleanType

RAIZ = "/Volumes/funcal/bronce/input_data"

# Convención de nombres. AJUSTAR a la del archivo real.
#   pieza suelta   : 711-021-A_REV00_SIM02_ENSAMBLE.STL
#   con colada     : ..._COLADA.STL   /  ..._ENSAMBLE - COLADA ....STL
#   racimo         : ..._RACIMO.STL   /  ..._X2.STL / ..._X3.STL
RE_COLADA   = re.compile(r"colada", re.I)
RE_RACIMO   = re.compile(r"racimo|_x[234]\b", re.I)
RE_ENSAMBLE = re.compile(r"ensamble", re.I)

def rol_de(nombre):
    if RE_RACIMO.search(nombre):   return "racimo"
    if RE_COLADA.search(nombre):   return "colada"
    if RE_ENSAMBLE.search(nombre): return "ensamble"
    return "pieza"


# COMMAND ----------

# MAGIC %md
# MAGIC ## Nivel 1 — descriptores geométricos

# COMMAND ----------

ESQUEMA = StructType([
    StructField("archivo", StringType()),
    StructField("clave_pieza", StringType()),
    StructField("rol", StringType()),                 # pieza | colada | racimo
    # --- tamaño y masa ---
    StructField("lx_m", DoubleType()), StructField("ly_m", DoubleType()),
    StructField("lz_m", DoubleType()), StructField("volumen_m3", DoubleType()),
    StructField("area_max_m2", DoubleType()),
    # --- topología / calidad ---
    StructField("n_triangulos", IntegerType()),
    StructField("n_cuerpos", IntegerType()),
    StructField("watertight", BooleanType()),
    StructField("n_cuerpos_congruentes", IntegerType()),   # multi-cavidad
    # --- forma ---
    StructField("compacidad", DoubleType()),          # V / bbox
    StructField("esbeltez", DoubleType()),            # L_mayor / sqrt(A_max)
    StructField("aspecto_xy", DoubleType()),          # Lx/Ly
    StructField("axisimetria", DoubleType()),         # 0 = revolución perfecta
    StructField("planitud_horizontal", DoubleType()), # Lz / max(Lx,Ly)
    # --- fondo (decide si la Fase X aplica) ---
    StructField("ondulacion_piso_x_m", DoubleType()),
    StructField("ondulacion_piso_y_m", DoubleType()),
    StructField("eje_fondo_plano", StringType()),     # X | Y | ninguno
    # --- perfil A(z) reducido ---
    StructField("az_centroide", DoubleType()),        # centro de masa en z/H
    StructField("az_dispersion", DoubleType()),       # 2º momento normalizado
    StructField("az_pico_rel", DoubleType()),         # z(A_max)/H
    StructField("az_salto_max", DoubleType()),        # mayor discontinuidad rel.
    StructField("az_salto_z", DoubleType()),          # dónde ocurre
    StructField("estado", StringType()),
])


def _perfil_area(mesh, n=120):
    """A(z) con unión real por rebanada. Devuelve (z_rel, A)."""
    zmin, zmax = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    span = zmax - zmin
    zs = np.linspace(zmin + 1e-4 * span, zmax - 1e-4 * span, n)
    A = np.zeros(n)
    for i, z in enumerate(zs):
        try:
            sec = mesh.section(plane_origin=[0, 0, float(z)], plane_normal=[0, 0, 1])
            if sec is None:
                continue
            plano, _ = sec.to_planar()
            u = _union_segura(plano.polygons_full)
            A[i] = float(u.area) if u is not None else 0.0
        except Exception:
            A[i] = np.nan
    A = pd.Series(A).interpolate(limit_direction="both").fillna(0.0).values
    return zs - zmin, A


def _ondulacion_piso(mesh, eje, n=60):
    """Rango de cotas del piso a lo largo de un eje horizontal.

    Es EL descriptor que decide si la Fase X (barrido en lámina) aplica: si el
    piso ondula más que h_ataque, el metal se embalsa en vez de barrer.
    """
    normal = [1, 0, 0] if eje == 0 else [0, 1, 0]
    lo, hi = float(mesh.bounds[0, eje]), float(mesh.bounds[1, eje])
    pisos = []
    for c in np.linspace(lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo), n):
        try:
            origen = [c, 0, 0] if eje == 0 else [0, c, 0]
            sec = mesh.section(plane_origin=origen, plane_normal=normal)
            if sec is None:
                continue
            plano, _ = sec.to_planar()
            u = _union_segura(plano.polygons_full)
            if u is not None:
                pisos.append(float(u.bounds[1]))
        except Exception:
            continue
    return float(max(pisos) - min(pisos)) if len(pisos) > 2 else float("nan")


def _axisimetria(mesh, n=20):
    """0 = sólido de revolución perfecto. Discrimina anillos/bridas/discos.

    Para cada rebanada, desviación relativa del radio del contorno mayor
    respecto a su centroide. Barato y muy discriminante: en las pruebas, un
    anillo dio 0.004 frente a 0.058 de una muela.
    """
    zmin, zmax = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    span = zmax - zmin
    vals = []
    for z in np.linspace(zmin + 0.05 * span, zmax - 0.05 * span, n):
        try:
            sec = mesh.section(plane_origin=[0, 0, float(z)], plane_normal=[0, 0, 1])
            if sec is None:
                continue
            plano, _ = sec.to_planar()
            polys = [p for p in plano.polygons_full if p is not None]
            if not polys:
                continue
            p = max(polys, key=lambda q: q.area)
            c = np.asarray(p.exterior.coords)
            r = np.linalg.norm(c - c.mean(axis=0), axis=1)
            if r.mean() > 1e-9:
                vals.append(float(r.std() / r.mean()))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else float("nan")


def _cuerpos_congruentes(mesh, tol=0.02):
    """Nº de cuerpos con volumen y extents casi idénticos -> multi-cavidad.

    Un racimo de 2-3 piezas iguales alimentadas por un canal común cumple la
    hipótesis de nivel único (A_total = n·A_pieza). Si los cuerpos NO son
    congruentes, los niveles divergen y el modelo 0D deja de aplicar.
    """
    try:
        partes = mesh.split(only_watertight=False)
    except Exception:
        return 1
    if len(partes) < 2:
        return 1
    firmas = []
    for p in partes:
        try:
            v = abs(float(p.volume))
        except Exception:
            v = float(np.prod(p.extents))
        firmas.append((v, tuple(np.round(np.sort(p.extents), 4))))
    firmas.sort(key=lambda s: -s[0])
    ref_v, ref_e = firmas[0]
    n = sum(1 for v, e in firmas
            if ref_v > 0 and abs(v - ref_v) / ref_v < tol
            and np.allclose(e, ref_e, rtol=tol))
    return int(n)


def _momentos_az(z, A):
    """Reduce el perfil A(z) a cuatro números comparables entre piezas."""
    H = z[-1] if z[-1] > 0 else 1.0
    w = A / max(A.sum(), 1e-12)
    zc = float(np.sum(w * z) / H)
    zd = float(np.sqrt(np.sum(w * (z / H - zc) ** 2)))
    zp = float(z[int(np.argmax(A))] / H)
    # mayor discontinuidad relativa entre nodos consecutivos
    Am = np.maximum(A[:-1], 1e-9)
    salto = np.abs(np.diff(A)) / Am
    k = int(np.argmax(salto))
    return zc, zd, zp, float(salto[k]), float(z[k] / H)


def descriptores(path, rol):
    """Fila completa de la huella para un STL."""
    fila = dict.fromkeys([f.name for f in ESQUEMA.fields])
    fila["archivo"] = path
    fila["clave_pieza"] = clave_pieza(path)
    fila["rol"] = rol
    try:
        m = trimesh.load_mesh(path)
        m.apply_scale(0.001)
        ext = m.extents
        if ext.max() < 0.01 or ext.max() > 20.0:
            fila["estado"] = f"ESCALA_SOSPECHOSA extents={np.round(ext,3).tolist()}"
            return fila

        fila["lx_m"], fila["ly_m"], fila["lz_m"] = map(float, ext)
        fila["n_triangulos"] = int(len(m.faces))
        fila["watertight"] = bool(m.is_watertight)
        try:
            fila["n_cuerpos"] = int(len(m.split(only_watertight=False)))
        except Exception:
            fila["n_cuerpos"] = 1
        fila["n_cuerpos_congruentes"] = _cuerpos_congruentes(m)

        z, A = _perfil_area(m)
        V = float(np.trapezoid(A, z))
        fila["volumen_m3"] = V
        fila["area_max_m2"] = float(A.max())
        fila["compacidad"] = V / float(np.prod(ext)) if np.prod(ext) > 0 else None
        L = max(ext[0], ext[1])
        fila["esbeltez"] = L / np.sqrt(max(A.max(), 1e-12))
        fila["aspecto_xy"] = float(ext[0] / ext[1]) if ext[1] > 0 else None
        fila["planitud_horizontal"] = float(ext[2] / L) if L > 0 else None
        fila["axisimetria"] = _axisimetria(m)

        ox = _ondulacion_piso(m, 0)
        oy = _ondulacion_piso(m, 1)
        fila["ondulacion_piso_x_m"], fila["ondulacion_piso_y_m"] = ox, oy
        # "Plano" en términos relativos: ondulación < 2 % de la longitud del eje.
        plano = []
        if np.isfinite(ox) and ox < 0.02 * ext[0]:
            plano.append("X")
        if np.isfinite(oy) and oy < 0.02 * ext[1]:
            plano.append("Y")
        fila["eje_fondo_plano"] = "+".join(plano) if plano else "ninguno"

        zc, zd, zp, s, sz = _momentos_az(z, A)
        fila["az_centroide"], fila["az_dispersion"] = zc, zd
        fila["az_pico_rel"], fila["az_salto_max"], fila["az_salto_z"] = zp, s, sz
        fila["estado"] = "OK"
    except Exception as e:
        # En un censo, un STL malo NO debe tumbar el job: se marca y sigue.
        fila["estado"] = f"ERROR {type(e).__name__}: {str(e)[:180]}"
    return fila


# COMMAND ----------

# MAGIC %md
# MAGIC ## Ejecución distribuida

# COMMAND ----------

def rol_de(nombre):
    if RE_RACIMO.search(nombre):
        return "racimo"
    if RE_COLADA.search(nombre):
        return "colada"
    return "pieza"


archivos = [f.path for f in dbutils.fs.ls(RAIZ) if f.name.lower().endswith(".stl")]
print(f"[*] {len(archivos)} STL encontrados en {RAIZ}")

df_in = spark.createDataFrame(
    pd.DataFrame({"path": [a.replace("dbfs:", "") for a in archivos]})
).repartition(min(64, max(1, len(archivos) // 20)))


@pandas_udf(ESQUEMA)
def udf_huella(paths: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([descriptores(p, rol_de(p)) for p in paths])


df_out = df_in.select(udf_huella(col("path")).alias("h")).select("h.*")
huella = df_out.toPandas()

ok = (huella["estado"] == "OK").sum()
print(f"[*] {ok}/{len(huella)} procesados correctamente")
for est in huella.loc[huella["estado"] != "OK", "estado"].head(10):
    print("    ", est)

(spark.createDataFrame(huella).write.format("delta")
 .mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable("funcal.silver.huella_geometrica"))
print("[*] Guardado en funcal.silver.huella_geometrica")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Nivel 2 — parámetros de colada por diferencia
# MAGIC
# MAGIC Requiere que la versión con colada y la versión suelta estén en el MISMO
# MAGIC sistema de coordenadas. Compruébalo antes: si los bounds de la pieza no
# MAGIC están contenidos en los del conjunto, cada archivo se re-origina al
# MAGIC exportar y este paso no es aplicable (habría que alinearlos primero).

# COMMAND ----------

def parametros_colada(path_conjunto, path_pieza, tol=0.02):
    """Extrae A_choke, vol_canales, ataques y z_ataque del sistema de colada.

    Método: cargar el conjunto, separar en cuerpos, descartar los que coinciden
    en volumen con la pieza suelta; lo que queda es la colada. Sobre ella:
      · A_choke   = mínima sección horizontal del bebedero
      · z_ataque  = cota (rel. al fondo de la pieza) donde la colada toca la pieza
      · h_ataque  = altura de esa zona de contacto
      · A_ataques = sección total en el contacto
    """
    mp = trimesh.load_mesh(path_pieza); mp.apply_scale(0.001)
    mc = trimesh.load_mesh(path_conjunto); mc.apply_scale(0.001)
    V_pieza = abs(float(mp.volume)) if mp.is_volume else float("nan")
    z_min_pieza = float(mp.bounds[0, 2])

    partes = mc.split(only_watertight=False)
    colada = [p for p in partes
              if not (np.isfinite(V_pieza) and p.is_volume
                      and abs(abs(p.volume) - V_pieza) / V_pieza < tol)]
    if not colada:
        return dict(estado="NO_SEPARABLE: ningún cuerpo distinto de la pieza")
    g = trimesh.util.concatenate(colada)

    zg0, zg1 = float(g.bounds[0, 2]), float(g.bounds[1, 2])
    zs = np.linspace(zg0 + 1e-4, zg1 - 1e-4, 300)
    Ag = np.zeros(len(zs))
    for i, z in enumerate(zs):
        try:
            sec = g.section(plane_origin=[0, 0, float(z)], plane_normal=[0, 0, 1])
            if sec is None:
                continue
            plano, _ = sec.to_planar()
            u = _union_segura(plano.polygons_full)
            Ag[i] = float(u.area) if u is not None else 0.0
        except Exception:
            Ag[i] = np.nan
    Ag = pd.Series(Ag).interpolate(limit_direction="both").fillna(0.0).values

    # El bebedero es el tramo superior de sección casi constante; el
    # estrangulador es su sección mínima.
    alto = zs > (zg0 + 0.5 * (zg1 - zg0))
    A_choke = float(np.min(Ag[alto])) if alto.any() else float(np.min(Ag[Ag > 0]))

    # Zona de ataques: donde la colada solapa en z con la pieza.
    solape = (zs >= float(mp.bounds[0, 2])) & (zs <= float(mp.bounds[1, 2]))
    if solape.any():
        z_at = zs[solape]
        z_ataque = float(z_at.max() - z_min_pieza)       # techo del ataque
        h_ataque = float(z_at.max() - z_at.min())
        A_ataques = float(np.max(Ag[solape]))
    else:
        z_ataque = h_ataque = A_ataques = float("nan")

    return dict(estado="OK",
                vol_canales_m3=float(np.trapezoid(Ag, zs)),
                a_choke_m2=A_choke,
                d_choke_equiv_m=float(2 * np.sqrt(A_choke / np.pi)),
                z_ataque_techo_m=z_ataque,
                z_ataque_centroide_m=z_ataque - h_ataque / 2 if np.isfinite(h_ataque) else float("nan"),
                h_ataque_m=h_ataque,
                a_ataques_m2=A_ataques,
                ratio_ataques=A_ataques / A_choke if A_choke > 0 else float("nan"),
                n_cuerpos_colada=len(colada))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Agrupamiento — descubrir las familias, no declararlas
# MAGIC
# MAGIC Se agrupa por descriptores y DESPUÉS se contrasta con la nomenclatura de
# MAGIC taller. Si los grupos coinciden con los nombres, la correspondencia queda
# MAGIC validada; si no, ese desacuerdo es el hallazgo: dos piezas que llamas
# MAGIC igual no se llenan igual.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

RASGOS = ["compacidad", "esbeltez", "axisimetria", "planitud_horizontal",
          "az_centroide", "az_dispersion", "az_pico_rel", "az_salto_max"]

d = huella[(huella["estado"] == "OK") & (huella["rol"] == "pieza")].copy()
X = d[RASGOS].astype(float)
X = X.fillna(X.median())
Xs = StandardScaler().fit_transform(X)

# Silueta para elegir k en vez de fijarlo a ojo
from sklearn.metrics import silhouette_score
for k in range(2, 9):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
    print(f"    k={k}  silueta={silhouette_score(Xs, lab):.3f}")

K = 5   # AJUSTAR según la silueta
d["familia"] = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(Xs)
p = PCA(n_components=2).fit_transform(Xs)
d["pca1"], d["pca2"] = p[:, 0], p[:, 1]

print(d.groupby("familia")[RASGOS + ["volumen_m3"]].median().round(3))

(spark.createDataFrame(d).write.format("delta")
 .mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable("funcal.silver.familias_piezas"))

# COMMAND ----------

piv = huella[huella.estado == "OK"].pivot_table(
    index="clave_pieza", columns="rol", values="volumen_m3", aggfunc="first")
piv["mazarota_pct"] = 100 * (1 - piv["pieza"] / piv["ensamble"])
piv["rendimiento_pct"] = 100 * piv["pieza"] / (piv["ensamble"] + piv["colada"])

for rol in ("pieza", "ensamble", "colada"):
    falta = piv[piv[rol].isna()]
    print(f"[!] {len(falta)} claves sin versión '{rol}'")