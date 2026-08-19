# Databricks notebook source
# MAGIC %md
# MAGIC # Núcleo compartido del modelo híbrido 1D-2D
# MAGIC
# MAGIC Guardar este notebook como `_core_llenado` en la **misma carpeta** que
# MAGIC `Tiempo_Llenado_Colada_General` y `K_general`.
# MAGIC
# MAGIC Ambos lo invocan con `%run ./_core_llenado`, de modo que la geometría y la
# MAGIC física son **idénticas** en simulación y calibración. Este archivo no lee
# MAGIC widgets ni escribe tablas: solo define funciones puras.

# COMMAND ----------

import numpy as np
import pandas as pd
try:
    import trimesh
    from shapely.geometry import box
    from shapely.ops import unary_union
except ImportError as e:
    raise ImportError(
        f"Falta '{e.name}' en la sesión.\n"
        "  · Serverless: panel Environment del notebook PADRE -> Add Dependency "
        "-> trimesh, shapely -> Apply.\n"
        "  · Cluster: librería de cluster, o '%pip install trimesh shapely' como "
        "PRIMERA celda del padre, antes del %run.\n"
        "  El entorno del núcleo no cuenta: %run se ejecuta en la sesión del padre."
    ) from e
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import DoubleType, ArrayType
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid
from scipy.signal import savgol_filter
from typing import Iterator

G = 9.81

# Umbral FÍSICO de área de ola. Por debajo de esto la sección es degenerada
# (plano tangente a la cara extrema) y el diámetro hidráulico se toma del
# fallback en vez de calcularse como 4A/P.  Ver nota (1) al final.
AREA_MIN_OLA = 1e-5      # m2
DH_MIN = 1e-3            # m, cota inferior dura de Dh

# ============================================================================
# 1. CACHÉ DE MALLA POR WORKER
# ============================================================================
# Antes se hacía load_mesh() en cada partición y en cada N del estudio de malla
# (~450 cargas). El diccionario vive en el proceso Python del worker, así que
# la malla se lee UNA vez por worker y se reutiliza en todas las particiones.
_MESH_CACHE = {}


def _get_mesh(stl_path, reparar):
    key = (stl_path, bool(reparar))
    if key not in _MESH_CACHE:
        m = trimesh.load_mesh(stl_path)
        m.apply_scale(0.001)                 # el CAD viene en mm
        if reparar:
            m.fill_holes()
            m.fix_normals()
        _MESH_CACHE[key] = m
    return _MESH_CACHE[key]


def cargar_malla_driver(stl_path, reparar=False):
    """Carga en el driver y valida escala/estanqueidad. Devuelve (bounds, diag)."""
    m = trimesh.load_mesh(stl_path)
    m.apply_scale(0.001)
    if reparar:
        m.fill_holes()
        m.fix_normals()
    ext = m.extents
    diag = {
        "extents_m": ext,
        "watertight": bool(m.is_watertight),
        "n_cuerpos": int(len(m.split(only_watertight=False))),
        "volumen_m3": float(abs(m.volume)) if m.is_volume else float("nan"),
    }
    if ext.max() < 0.01 or ext.max() > 20.0:
        raise ValueError(
            f"[ESCALA] Tras apply_scale(0.001) la pieza mide {np.round(ext,4)} m. "
            "Comprueba si el STL ya venía en metros."
        )
    b = m.bounds.copy()
    del m
    return b, diag


# ============================================================================
# 2. UDFs DE SECCIONADO
# ============================================================================
TOLERAR_FALLOS = False   # True solo para mallas que sabes imperfectas
def _union_segura(polys):
    """Unión robusta de los contornos de un corte.

    Tres estrategias en cascada porque `union_all` (ufunc create_collection)
    falla con combinaciones antiguas de shapely/numpy:
      1. union_all sobre array dtype=object
      2. unión binaria acumulativa (usa el ufunc `union`, otro camino en C)
      3. lo mismo tras buffer(0) para sanear geometrías inválidas
    Devuelve None si no queda nada válido (área real ~0, no es un fallo).
    """
    from functools import reduce

    limpios = [p for p in polys
               if p is not None
               and getattr(p, "geom_type", "") in ("Polygon", "MultiPolygon")
               and not p.is_empty and p.area > 1e-12]
    if not limpios:
        return None
    if len(limpios) == 1:
        g = limpios[0]
        return g if g.is_valid else g.buffer(0)

    arr = np.empty(len(limpios), dtype=object)
    arr[:] = limpios
    try:
        return unary_union(arr)
    except Exception:
        pass
    try:
        return reduce(lambda a, b: a.union(b), limpios)
    except Exception:
        pass
    saneados = [g if g.is_valid else g.buffer(0) for g in limpios]
    return reduce(lambda a, b: a.union(b), saneados)
    
def _perimetro_mojado(geom, v_libre, tol=1e-7):
    """Perímetro de la ola EXCLUYENDO la cuerda de superficie libre.

    El estudio define P_mojado sin la cara superior (fricción nula contra el
    gas). `geom.length` la incluye, y el `/2.0` del código antiguo solo acierta
    en el límite W>>h. Aquí se descuentan los tramos que yacen exactamente en
    v = v_libre, que son los introducidos por el recorte de la banda.
    Los contornos interiores (almas/machos) sí cuentan: están mojados.
    """
    total = 0.0
    partes = geom.geoms if hasattr(geom, "geoms") else [geom]
    for p in partes:
        if p.is_empty or p.geom_type != "Polygon":
            continue
        for ring in [p.exterior, *p.interiors]:
            c = np.asarray(ring.coords)
            if len(c) < 2:
                continue
            d = np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1]))
            libre = (np.abs(c[:-1, 1] - v_libre) < tol) & (np.abs(c[1:, 1] - v_libre) < tol)
            total += float(d[~libre].sum())
    return total


def make_udf_area_z(stl_path, reparar):
    """Área de la sección horizontal en la cota z. NaN si el corte falla."""

    @pandas_udf(DoubleType())
    def _udf(z_batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
        mesh = _get_mesh(stl_path, reparar)
        for z_batch in z_batches:
            out = []
            for z in z_batch:
                a = np.nan
                try:
                    sec = mesh.section(plane_origin=[0, 0, float(z)], plane_normal=[0, 0, 1])
                    a = 0.0                      # sin sección = área real 0
                    if sec is not None:
                        plano, _ = sec.to_planar()
                        u = _union_segura(plano.polygons_full)
                        a = float(u.area) if u is not None else 0.0
                except Exception as e:
                    if not TOLERAR_FALLOS:
                        raise RuntimeError(
                            f"Corte Z fallido en z={z:.6f}: {type(e).__name__}: {e}. "
                            "Si es 'unable to recover polygon', el sólido tiene cuerpos "
                            "interpenetrados o caras coplanares conflictivas: fusiona en "
                            "el CAD y reexporta como *_FUSED.STL."
                        ) from e
                    a = np.nan
                out.append(a)
            yield pd.Series(out, dtype="float64")

    return _udf

def make_udf_geom_x(stl_path, reparar, h_ataque, eje_x):
    """Área y perímetro mojado de la ola en la estación de barrido.

    Devuelve [area, perim_mojado]; [NaN, NaN] si el corte falla.
    """
    normal = [1, 0, 0] if eje_x else [0, 1, 0]
    # Proyección al plano 2D con U = el otro eje horizontal, V = Z real.
    # La 3ª fila lleva la traslación -coord para que la sección caiga en w=0;
    # sin ella to_planar(check=True) rechaza todo corte que no pase por el origen.
    if eje_x:
        BASE = np.array([[0., 1., 0., 0.],
                         [0., 0., 1., 0.],
                         [1., 0., 0., 0.],
                         [0., 0., 0., 1.]])
    else:
        BASE = np.array([[1., 0., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 0., 1.]])

    @pandas_udf(ArrayType(DoubleType()))
    def _udf(coord_batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
        mesh = _get_mesh(stl_path, reparar)
        for coord_batch in coord_batches:
            out = []
            for coord in coord_batch:
                coord = float(coord)
                area, perim = np.nan, np.nan
                try:
                    origen = [coord, 0, 0] if eje_x else [0, coord, 0]
                    sec = mesh.section(plane_origin=origen, plane_normal=normal)
                    area, perim = 0.0, 0.0       # estación vacía = área real 0
                    if sec is not None:
                        M = BASE.copy()
                        M[2, 3] = -coord
                        plano, _ = sec.to_planar(to_2D=M, check=False)
                        u = _union_segura(plano.polygons_full)
                        if u is not None:
                            min_u, piso, max_u, _ = u.bounds
                            v_libre = piso + h_ataque
                            pad = 0.05 * max(1.0, max_u - min_u)
                            banda = box(min_u - pad, piso, max_u + pad, v_libre)
                            ola = u.intersection(banda)
                            if not ola.is_empty:
                                area = float(ola.area)
                                perim = _perimetro_mojado(ola, v_libre)
                except Exception as e:
                    if not TOLERAR_FALLOS:
                        raise RuntimeError(
                            f"Corte X fallido en coord={coord:.6f}: {type(e).__name__}: {e}. "
                            "Comprueba que el sólido esté fusionado."
                        ) from e
                    area, perim = np.nan, np.nan
                out.append([area, perim])
            yield pd.Series(out)

    return _udf


# ============================================================================
# 3. EXTRACCIÓN DE PERFILES
# ============================================================================
def _reparar_serie(vals, etiqueta, umbral_abortar=0.20):
    s = pd.Series(vals, dtype="float64")
    n_fallos = int(s.isna().sum())
    frac = n_fallos / max(len(s), 1)
    if frac > umbral_abortar:
        raise RuntimeError(
            f"[GEOMETRÍA] {etiqueta}: {n_fallos}/{len(s)} cortes fallidos "
            f"({100*frac:.1f} %). Interpolar esto no produciría un resultado "
            "físico. Revisa dependencias (networkx), la ruta del STL y la "
            "estanqueidad de la malla."
        )
    if n_fallos:
        s = s.interpolate(limit_direction="both")
        print(f"    [!] {etiqueta}: {n_fallos} cortes fallidos interpolados "
              f"({100*frac:.1f} %)")
    return s.fillna(0.0).values, n_fallos


def _nodos(a, b, n):
    """Nodos interiores + extremos exactos.

    Cortar justo en el bound devuelve una sección degenerada (área falsa 0 o
    lazo de área nula con perímetro no nulo). Se muestrea en el interior y se
    extienden los extremos con el valor vecino sobre un intervalo de 1e-4 L.
    """
    eps = 1e-4 * (b - a)
    return np.linspace(a + eps, b - eps, n)


def _spark_perfil(coords, columna, udf, etiqueta):
    n_part = max(1, min(64, len(coords) // 25))
    df = spark.createDataFrame(pd.DataFrame({columna: coords})).repartition(n_part)
    res = df.withColumn("val", udf(col(columna))).orderBy(columna).toPandas()
    return res


def extraer_perfiles(stl_path, bounds, num_cortes, h_ataque, reparar=False,
                     savgol=False, con_fase_x=True, eje_forzado=None):
    """Devuelve los vectores geométricos cacheados (no dependen de K)."""
    x_min, x_max = bounds[:, 0]
    y_min, y_max = bounds[:, 1]
    z_min, z_max = bounds[:, 2]
    lx, ly = x_max - x_min, y_max - y_min

    # Eje de avance del frente. NO es necesariamente X: por defecto se toma el
    # más largo de X/Y, pero el criterio físico es el eje PERPENDICULAR a la
    # línea de ataques, que el código no puede conocer (no ve el STL de colada).
    # Elegir mal el eje cambia K_fricción en un factor 2 sin ningún síntoma.
    if eje_forzado in ("X", "x"):
        eje_x = True
    elif eje_forzado in ("Y", "y"):
        eje_x = False
    else:
        eje_x = lx >= ly
        if max(lx, ly) > 0 and abs(lx - ly) / max(lx, ly) < 0.05:
            print(f"    [!] Lx={lx:.4f} m y Ly={ly:.4f} m difieren menos del 5 %. "
                  "La elección automática de eje es frágil: puede voltear entre "
                  "revisiones del CAD. Fija el eje explícitamente.")

    L_molde = lx if eje_x else ly
    c_min, c_max = (x_min, x_max) if eje_x else (y_min, y_max)

    def _suavizar(v, piso):
        if not savgol or len(v) < 13:
            return np.maximum(v, piso)
        w = max(5, int(len(v) * 0.05))
        w = w if w % 2 else w + 1
        w = min(w, len(v) - 1 if (len(v) - 1) % 2 else len(v) - 2)
        return np.maximum(savgol_filter(v, w, 3), piso)

    # ---- eje vertical -------------------------------------------------
    zc = _nodos(z_min, z_max, num_cortes)
    res = _spark_perfil(zc, "z_coord", make_udf_area_z(stl_path, reparar), "eje Z")
    az, fz_z = _reparar_serie(res["val"].values, "eje Z")
    z_vals = np.concatenate([[0.0], res["z_coord"].values - z_min, [z_max - z_min]])
    areas_z = np.concatenate([[az[0]], az, [az[-1]]])
    areas_z = _suavizar(areas_z, 0.0)

    out = dict(z_vals=z_vals, areas_z=areas_z, L_molde=float(L_molde),
               eje_x=bool(eje_x), z_span=float(z_max - z_min),
               fallos_z=fz_z, fallos_x=0)

    # ---- eje horizontal -----------------------------------------------
    if con_fase_x:
        hc = _nodos(c_min, c_max, num_cortes)
        res = _spark_perfil(hc, "coord_horiz",
                            make_udf_geom_x(stl_path, reparar, h_ataque, eje_x), "eje X/Y")
        a_raw = res["val"].apply(lambda v: v[0]).values
        p_raw = res["val"].apply(lambda v: v[1]).values
        a_h, f1 = _reparar_serie(a_raw, "eje X/Y (área)")
        p_h, f2 = _reparar_serie(p_raw, "eje X/Y (perímetro)")
        h_vals = np.concatenate([[0.0], res["coord_horiz"].values - c_min, [L_molde]])
        areas_h = _suavizar(np.concatenate([[a_h[0]], a_h, [a_h[-1]]]), 0.0)
        perim_h = _suavizar(np.concatenate([[p_h[0]], p_h, [p_h[-1]]]), 0.0)
        out.update(h_vals=h_vals, areas_h=areas_h, perim_h=perim_h,
                   fallos_x=max(f1, f2))
    out["h_ataque_usada"] = float(h_ataque)
    out["eje_barrido"] = "X" if eje_x else "Y"
    out["eje_forzado"] = eje_forzado or "Auto"
    return out


# ============================================================================
# 4. FÍSICA
# ============================================================================
def k_friccion(perfiles, f_arena, h_ataque, dinamico=True):
    """K acumulado por barrido del lecho (Fase 2 del estudio).

    Dh de referencia = 4*h_ataque: diámetro hidráulico de un canal ancho de
    calado h_ataque (P_mojado -> W cuando W >> h). Se usa como modelo completo
    en la Opción A (estático) y como valor de reemplazo para las secciones
    degeneradas en la Opción B. No es un parámetro libre: fijarlo aparte de
    h_ataque solo permite declarar dos calados incompatibles.
    """
    if "areas_h" not in perfiles:
        return 0.0, None
    dh_ref = 4.0 * h_ataque
    if not dinamico:
        Kf = f_arena * (perfiles["h_vals"] / dh_ref)
        return float(Kf[-1]), Kf

    A = perfiles["areas_h"]
    P_mojado = np.maximum(perfiles["perim_h"], 1e-9)
    Dh = np.where(A < AREA_MIN_OLA, dh_ref, 4.0 * A / P_mojado)
    Dh = np.clip(Dh, DH_MIN, 5.0 * dh_ref)
    Kf = cumulative_trapezoid(y=f_arena / Dh, x=perfiles["h_vals"], initial=0.0)
    return float(Kf[-1]), Kf


def integrando_z(perfiles, H_total, z_ataque):
    """f_z = A(z)/sqrt(H_ef(z)); no depende de K."""
    z = perfiles["z_vals"]
    if perfiles["z_span"] >= H_total:
        raise ValueError(
            f"[FÍSICA] La pieza mide {perfiles['z_span']:.3f} m y H_total es "
            f"{H_total:.3f} m. El metal no puede alcanzar la cota máxima; el "
            "modelo de carga metalostática no aplica. Revisa Hc/h_caida."
            "z_ataque es la cota del CENTROIDE del ataque sobre el fondo de la pieza"
            "(z = 0 es el punto más bajo de la malla, no el origen del STL). El"
            "centroide, y no el techo ni el fondo, porque la carga efectiva de un"
            "orificio se mide a su centro de área, y porque es el punto medio de la"
            "transición entre descarga libre y descarga sumergida."
        )
    carga = np.where(z < z_ataque, H_total - z_ataque, H_total - z)
    carga = np.maximum(carga, 1e-6)
    return perfiles["areas_z"] / np.sqrt(carga), np.sqrt(carga)


def tiempos_llenado(perfiles, H_total, z_ataque, A_choke, K_sistema, K_fric,
                    metodo="Simpson"):
    """Tiempos de la fase vertical. t = C_z * I, con C_z = sqrt(1+K_z_ef)/(A*sqrt(2g))."""
    f_z, raiz = integrando_z(perfiles, H_total, z_ataque)
    z = perfiles["z_vals"]
    K_z_ef = K_sistema + K_fric
    C_z = np.sqrt(1.0 + K_z_ef) / (A_choke * np.sqrt(2 * G))
    integra = simpson if metodo == "Simpson" else trapezoid

    i = int(np.searchsorted(z, z_ataque))
    if i >= len(z):
        t_est, t_tr = integra(y=f_z, x=z), 0.0
    elif i == 0:
        t_est, t_tr = 0.0, integra(y=f_z, x=z)
    else:
        t_est = integra(y=f_z[:i + 1], x=z[:i + 1])
        t_tr = integra(y=f_z[i:], x=z[i:])

    curva = cumulative_trapezoid(y=f_z, x=z, initial=0.0) * C_z
    return dict(t_est=float(t_est * C_z), t_trans=float(t_tr * C_z),
                t_total=float((t_est + t_tr) * C_z), K_z_ef=float(K_z_ef),
                C_z=float(C_z), I=float(t_est + t_tr), f_z=f_z, raiz=raiz,
                z_curva=z, t_curva=curva)


def t_cebado(vol_canales, A_choke, H_total, K_sistema):
    """Purgado del cebado, coherente con la pérdida del sistema.

    El script antiguo usaba Q = A*sqrt(g*H), que equivale a K=0 justo en la
    variable que se está calibrando. La velocidad real durante el cebado es
    sqrt(2gH/(1+K)).
    """
    return float(vol_canales * np.sqrt(1.0 + K_sistema) /
                 (A_choke * np.sqrt(2 * G * H_total)))


def calibrar_K(t_planta, perfiles, H_total, z_ataque, A_choke, K_fric,
               vol_canales, metodo="Simpson"):
    """Calibración en FORMA CERRADA. No hace falta root_scalar ni bracket.

    t(K) = base*sqrt(1+K+Kf) + c*sqrt(1+K)  con base e Kf independientes de K.
    Sustituyendo s = sqrt(1+K) queda una cuadrática en s:
        (base^2 - c^2) s^2 + 2 t_p c s + (base^2 Kf - t_p^2) = 0
    """
    f_z, _ = integrando_z(perfiles, H_total, z_ataque)
    z = perfiles["z_vals"]
    integra = simpson if metodo == "Simpson" else trapezoid
    I = float(integra(y=f_z, x=z))
    base = I / (A_choke * np.sqrt(2 * G))
    c = vol_canales / (A_choke * np.sqrt(2 * G * H_total))

    t_min = base * np.sqrt(1.0 + K_fric) + c        # K_sistema = 0
    if t_planta <= t_min:
        raise ValueError(
            f"[RANGO] t_planta = {t_planta:.2f} s es inferior al mínimo físico "
            f"{t_min:.2f} s (K_sistema = 0, K_fricción = {K_fric:.3f}). "
            f"Con A_choke = {A_choke:.6f} m2 ningún K puede dar ese tiempo: "
            "el estrangulador es demasiado pequeño o el tiempo de planta es erróneo."
        )

    a = base ** 2 - c ** 2
    b = 2.0 * t_planta * c
    d = base ** 2 * K_fric - t_planta ** 2
    s = (-b + np.sqrt(b * b - 4 * a * d)) / (2 * a)
    K = float(s * s - 1.0)
    return dict(K_sistema=K, K_z_ef=K + K_fric, I=I, base=base,
                t_cebado=float(c * s), t_min_fisico=float(t_min))

# COMMAND ----------

def diagnostico_barrido(perfiles, H_total, z_ataque, A_choke, K_z_ef):
    """Barrido del piso como DIAGNÓSTICO, nunca como sumando.

    Durante el barrido el nivel está bajo z_ataque, así que la carga es
    constante y t = V_ola/Q. Ese tiempo ya está dentro de t_estacionario:
    la integral vertical cubre el mismo volumen. Lo útil es compararlo con
    lo que tarda el nivel en superar h_ataque: si el frente es más lento,
    el llenado inicial está dominado por el frente y la hipótesis de
    superficie libre horizontal no aplica en ese tramo.
    """
    if "areas_h" not in perfiles:
        return {}
    carga = H_total - z_ataque
    Q = A_choke * np.sqrt(2 * G * carga / (1.0 + K_z_ef))
    V_ola = float(trapezoid(perfiles["areas_h"], perfiles["h_vals"]))
    t_barrido = V_ola / Q

    z, A = perfiles["z_vals"], perfiles["areas_z"]
    Vz = cumulative_trapezoid(A, z, initial=0.0)
    h_a = perfiles.get("h_ataque_usada", None)
    return dict(
        volumen_ola_m3=V_ola,
        t_barrido_piso_s=float(t_barrido),
        z_nivel_al_fin_barrido_m=float(np.interp(V_ola, Vz, z)),
        frente_domina_inicio=bool(t_barrido > float(np.interp(h_a, z, Vz) / Q)) if h_a else None,
    )

# COMMAND ----------

CORE_VERSION = "2026-08-17.eje-forzado"
print(f"[core] _core_llenado cargado: {CORE_VERSION}")