# Databricks notebook source
# MAGIC %md
# MAGIC # Tiempo de llenado — modelo híbrido 1D-2D (v2)
# MAGIC Requiere `_core_llenado` en la misma carpeta.

# COMMAND ----------

# MAGIC %run ./_core_llenado

# COMMAND ----------

_ESPERADO = "2026-08-17.eje-forzado"
if globals().get("CORE_VERSION") != _ESPERADO:
    raise RuntimeError(
        f"Núcleo cargado: '{globals().get('CORE_VERSION')}', esperado '{_ESPERADO}'. "
        "Vuelve a ejecutar la celda '%run ./_core_llenado'.")

# COMMAND ----------

import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# 1. WIDGETS
# ============================================================================
dbutils.widgets.text("1_archivo_stl", "711-021-A_REV00_SIM02_ENSAMBLE.STL", "1. Archivo STL")
dbutils.widgets.dropdown("2_sistema", "Copa", ["Balsa", "Copa"], "2. Sistema Vaciado")
dbutils.widgets.dropdown("3_metodo", "Simpson", ["Simpson", "Trapezoidal"], "3. Método Numérico")
dbutils.widgets.text("4_Hc", "0.666", "4. Carga Inicial Hc (m)")
dbutils.widgets.text("5_diametro_inferior", "0.070", "5. Diámetro choke d_inf (m)")
dbutils.widgets.text("6_zataque", "0.046456",
                     "6. Cota Z del EJE de ataques (m sobre el fondo de la pieza)")
dbutils.widgets.text("7_h_caida", "0.250", "7. Caída Cuchara (m) [Solo Copa]")
dbutils.widgets.text("8_Ksistema", "", "8. K_sistema (Vacío = leer calibración)")
dbutils.widgets.text("9_N_inicial", "50", "9. Malla Inicial (N)")
dbutils.widgets.text("10_tolerancia", "0.1", "10. Tolerancia Parada (%)")
dbutils.widgets.text("11_Z_boca", "0.566", "11. Cota Tope de Arena Z_boca (m)")
dbutils.widgets.text("12_Z_base", "0.122", "12. Cota Base Bebedero Z_base (m)")
dbutils.widgets.dropdown("13_autoreparar", "No", ["No", "Si"], "13. Auto-Reparar STL")
dbutils.widgets.dropdown("14_filtro_savgol", "No", ["No", "Si"], "14. Filtro Savitzky-Golay")
dbutils.widgets.dropdown("15_modelo_hibrido", "Si", ["No", "Si"], "15. Activar Fase X (Híbrido)")
dbutils.widgets.text("16_f_arena", "0.045", "16. Factor Fricción Arena (f)")
dbutils.widgets.dropdown("17_Dh_dinamico", "Si", ["No", "Si"], "17. D_h Dinámico [Opción B]")
dbutils.widgets.text("18_h_ataque", "0.018", "18. Altura física del ataque (m)")
# NUEVO: confirmaciones consecutivas bajo tolerancia (el error NO es monótono)
dbutils.widgets.text("19_confirmaciones", "2", "19. Mallas consecutivas bajo tolerancia")

dbutils.widgets.text("20_area_ataques", "0.00758", "20. Área total de ataques (m2)")
dbutils.widgets.text("21_z_tope_pieza", "0.148", "21. Cota Z tope de la PIEZA sin mazarotas (m)")
dbutils.widgets.text("22_v_critica", "0.5", "22. Velocidad crítica de la aleación (m/s)")

dbutils.widgets.text("23_densidad", "8800", "23. Densidad de la aleación (kg/m3)")

dbutils.widgets.text("24_volumen_cad", "0", "24. Volumen del CAD (m3, 0 = no comprobar)")

dbutils.widgets.dropdown("25_eje_barrido", "Auto", ["Auto", "X", "Y"],
                         "25. Eje de avance del frente")

stl_name = dbutils.widgets.get("1_archivo_stl").strip()
sistema = dbutils.widgets.get("2_sistema")
metodo = dbutils.widgets.get("3_metodo")
STL_PATH = f"/Volumes/funcal/bronce/input_data/{stl_name}"
nombre_limpio = stl_name.replace(".STL", "").replace(".stl", "").replace("-", "_").lower()

Hc = float(dbutils.widgets.get("4_Hc"))
d_inf = float(dbutils.widgets.get("5_diametro_inferior"))
A_choke = np.pi * d_inf ** 2 / 4
z_ataque = float(dbutils.widgets.get("6_zataque"))
N_INICIAL = int(dbutils.widgets.get("9_N_inicial"))
TOLERANCIA = float(dbutils.widgets.get("10_tolerancia"))
N_MAXIMO = 3200
Z_boca = float(dbutils.widgets.get("11_Z_boca"))
Z_base = float(dbutils.widgets.get("12_Z_base"))
AUTO_REPARAR = dbutils.widgets.get("13_autoreparar") == "Si"
FILTRO_SAVGOL = dbutils.widgets.get("14_filtro_savgol") == "Si"
MODELO_HIBRIDO = dbutils.widgets.get("15_modelo_hibrido") == "Si"
f_arena = float(dbutils.widgets.get("16_f_arena"))
DH_DINAMICO = dbutils.widgets.get("17_Dh_dinamico") == "Si"
h_ataque = float(dbutils.widgets.get("18_h_ataque"))
N_CONFIRM = max(1, int(dbutils.widgets.get("19_confirmaciones")))

A_ataques = float(dbutils.widgets.get("20_area_ataques"))
z_tope_pieza = float(dbutils.widgets.get("21_z_tope_pieza"))
V_CRITICA = float(dbutils.widgets.get("22_v_critica"))

DENSIDAD = float(dbutils.widgets.get("23_densidad"))

V_CAD = float(dbutils.widgets.get("24_volumen_cad"))   # 0 = sin comprobar

EJE_BARRIDO = dbutils.widgets.get("25_eje_barrido")

h_caida = float(dbutils.widgets.get("7_h_caida")) if sistema == "Copa" else 0.0
H_total = Hc + h_caida

# ---- K_sistema: widget, o el último calibrado para esta pieza --------------
TABLA_CALIB = "funcal.silver.calibracion_k_sistema"
k_input = dbutils.widgets.get("8_Ksistema").strip()
if k_input:
    K_sistema, origen_k = float(k_input), "widget"
    fila = None
    print("    [i] K_sistema forzado por widget: no se valida contra la calibración.")
else:
    try:
        fila = spark.sql(
            f"SELECT k_sistema, dh_dinamico_activo, f_arena, h_ataque_m, "
            f"achoke_m2, eje_barrido, modelo_hibrido_activo, z_ataque_m "
            f"FROM {TABLA_CALIB} WHERE pieza_stl = '{stl_name}' "
            "ORDER BY fecha_calibracion DESC LIMIT 1").collect()[0]
        K_sistema, origen_k = float(fila[0]), "tabla de calibración"
    except Exception:
        fila = None
        K_sistema = 4.5 if sistema == "Copa" else 2.5
        origen_k = "DEFECTO (sin calibrar — el resultado es orientativo)"

    if fila is not None:
        if fila[1] is not None and bool(fila[1]) != DH_DINAMICO:
            raise ValueError(
                f"[CONFIG] El K_sistema guardado se calibró con Dh_dinamico="
                f"{bool(fila[1])} y vas a simular con {DH_DINAMICO}.")
        for idx, (nom, val) in enumerate(
                [("f_arena", f_arena), ("h_ataque", h_ataque),
                 ("A_choke", A_choke)], start=2):
            ref = fila[idx]
            if ref is not None and abs(float(ref) - val) > 1e-9:
                raise ValueError(
                    f"[CONFIG] {nom} calibrado = {float(ref):.6g}, "
                    f"simulación = {val:.6g}.")
        if fila[7] is not None and abs(float(fila[7]) - z_ataque) > 1e-9:
            raise ValueError(
                f"[CONFIG] z_ataque calibrado = {float(fila[7]):.6g}, "
                f"simulación = {z_ataque:.6g}. Cambia la integral y por tanto "
                "el K. Recalibra o iguala la cota (¿convención de centroide?).")

print(f"[*] K_sistema: {K_sistema:.4f} ({origen_k})")

# ---- validaciones que antes no existían -----------------------------------
if H_total <= Z_boca:
    raise ValueError(f"[GEOMETRÍA] H_total ({H_total:.3f}) debe superar Z_boca ({Z_boca:.3f}).")
d_sup = d_inf * ((H_total - Z_base) / (H_total - Z_boca)) ** 0.25

bounds, diag = cargar_malla_driver(STL_PATH, AUTO_REPARAR)

# ---- eje de avance del frente ---------------------------------------------
lx = bounds[1, 0] - bounds[0, 0]
ly = bounds[1, 1] - bounds[0, 1]
if EJE_BARRIDO in ("X", "Y"):
    eje_efectivo = EJE_BARRIDO
else:
    eje_efectivo = "X" if lx >= ly else "Y"
    if max(lx, ly) > 0 and abs(lx - ly) / max(lx, ly) < 0.05:
        print(f"    [!] Lx={lx:.4f} m y Ly={ly:.4f} m difieren menos del 5 %: la "
              "elección automática de eje es frágil. Fíjalo con el widget 25.")

# L_barrido DEBE derivar de eje_efectivo, no de max(lx, ly): si se fuerza el eje
# corto, la compuerta de la Fase X quedaría decidida con la longitud equivocada.
L_barrido = lx if eje_efectivo == "X" else ly
K_fric_estimado = f_arena * L_barrido / (4 * h_ataque)
CON_FASE_X = MODELO_HIBRIDO and K_fric_estimado > 0.15

# ---- guarda de coherencia del eje (necesita bounds) ------------------------
if fila is not None:
    if fila[5] is not None and str(fila[5]) != eje_efectivo:
        raise ValueError(
            f"[CONFIG] El K_sistema guardado se calibró con barrido en {fila[5]} "
            f"y vas a simular con {eje_efectivo}. K_fricción puede cambiar en un "
            "factor 2. Recalibra o fija el mismo eje con el widget 25.")
    if fila[6] is not None and bool(fila[6]) != CON_FASE_X:
        raise ValueError(
            f"[CONFIG] El K_sistema guardado se calibró con Fase X={bool(fila[6])} "
            f"y vas a simular con {CON_FASE_X}. K_fricción no será el mismo y el "
            "descuadre ronda el 7 %. Recalibra o iguala el modo.")

# ---- resumen de configuración ---------------------------------------------
print(f"[*] {stl_name} | Sistema: {sistema} | Integración: {metodo}")
print(f"[*] Extents: {np.round(diag['extents_m'],4)} m | cuerpos: {diag['n_cuerpos']} | "
      f"watertight: {diag['watertight']}")
if not diag["watertight"]:
    print("    [!] Malla no estanca: algunos cortes pueden fallar (se interpolan y se reportan).")
if diag["n_cuerpos"] > 1:
    print(f"    [!] {diag['n_cuerpos']} cuerpos disjuntos. Se integra la UNIÓN "
          "(unary_union), no la suma de polígonos. Verifica que todos deban llenarse.")
print(f"[*] Eje de barrido: {eje_efectivo} (L = {L_barrido:.3f} m, origen: {EJE_BARRIDO})")
print(f"[*] K_fricción estimado: {K_fric_estimado:.3f} | "
      f"Fase X: {'activa' if CON_FASE_X else 'DESACTIVADA'} | Dh dinámico: {DH_DINAMICO}")
if MODELO_HIBRIDO and not CON_FASE_X:
    print("    [i] Fase X desactivada automáticamente: la fricción estimada es "
          "despreciable (<0.15). K_sistema absorberá lo poco que haya.")
print(f"[*] H_total: {H_total:.3f} m | K_sistema: {K_sistema:.3f} ({origen_k}) | "
      f"d_sup teórico: {d_sup*1000:.1f} mm")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Estudio de independencia de malla

# COMMAND ----------

def simular(num_cortes):
    perf = extraer_perfiles(STL_PATH, bounds, num_cortes, h_ataque,
                            reparar=AUTO_REPARAR, savgol=FILTRO_SAVGOL,
                            con_fase_x=CON_FASE_X,
                            eje_forzado=None if EJE_BARRIDO == "Auto" else EJE_BARRIDO)
    K_fric, _ = k_friccion(perf, f_arena, h_ataque, dinamico=DH_DINAMICO)
    r = tiempos_llenado(perf, H_total, z_ataque, A_choke, K_sistema, K_fric, metodo)
    r["perfiles"] = perf
    r["K_fric"] = K_fric
    return r

print("\n--- ESTUDIO DE INDEPENDENCIA DE MALLA ---")
n_actual = N_INICIAL
hist_t, hist_e, hist_n, hist_k = [], [], [], []

r = simular(n_actual)
t_prev = r["t_total"]
mejor = r
hist_t.append(t_prev); hist_e.append(np.nan); hist_n.append(n_actual); hist_k.append(r["K_fric"])
print(f"Iteración 1 | N: {n_actual:5} | t: {t_prev:9.4f} s | K_fric: {r['K_fric']:8.4f} | Error: ---")

error_relativo = np.inf
convergido = False
seguidas = 0

while n_actual < N_MAXIMO:
    n_actual *= 2
    r = simular(n_actual)
    t_curr = r["t_total"]
    if t_curr <= 0 or not np.isfinite(t_curr):
        raise RuntimeError(
            f"[MODELO] t_total = {t_curr} en N={n_actual}. La extracción "
            "geométrica no devolvió áreas útiles; revisa los avisos de cortes "
            "fallidos más arriba."
        )
    error_relativo = abs((t_curr - t_prev) / t_curr) * 100
    hist_t.append(t_curr); hist_e.append(error_relativo)
    hist_n.append(n_actual); hist_k.append(r["K_fric"])
    mejor = r
    print(f"Iteración {len(hist_n)} | N: {n_actual:5} | t: {t_curr:9.4f} s | "
          f"K_fric: {r['K_fric']:8.4f} | Error: {error_relativo:.4f} %")

    # El error NO es monótono (el escalón geométrico introduce ruido de
    # discretización), así que se exigen N_CONFIRM mallas seguidas bajo tolerancia.
    seguidas = seguidas + 1 if error_relativo < TOLERANCIA else 0
    if seguidas >= N_CONFIRM:
        convergido = True
        print(f"\n[ÉXITO] Convergencia en N = {n_actual} "
              f"({N_CONFIRM} mallas seguidas bajo {TOLERANCIA} %).")
        break
    t_prev = t_curr

if not convergido:
    print(f"\n[ALERTA] Malla máxima N={N_MAXIMO} sin convergencia estricta "
          f"(último error {error_relativo:.4f} %). Se usa la última iteración.")

# Diagnóstico que antes no existía: K_fric dividiéndose por 2 al duplicar N
# es la firma del nodo degenerado, no una propiedad física.
if len(hist_k) > 2 and hist_k[-1] > 1e-9:
    ratio = hist_k[-2] / hist_k[-1]
    if 1.7 < ratio < 2.3:
        print(f"[!] K_fricción se divide por {ratio:.2f} al duplicar N: sigue habiendo "
              "un nodo singular en la Fase X. Revisa AREA_MIN_OLA en el núcleo.")
  
perf = mejor["perfiles"]
z_opt, A_opt = perf["z_vals"], perf["areas_z"]
denom_opt, fz_opt = mejor["raiz"], mejor["f_z"]
L_molde, K_opt = perf["L_molde"], mejor["K_z_ef"]
tiempo_total_final = mejor["t_total"]

diag_barrido = diagnostico_barrido(perf, H_total, z_ataque, A_choke, K_opt)
t_horiz = diag_barrido.get("t_barrido_piso_s", 0.0)
if diag_barrido.get("frente_domina_inicio"):
    print(f"[!] El frente tarda {t_horiz:.2f} s en recorrer el molde, más que lo que "
          "el nivel tarda en superar h_ataque. El inicio del llenado está dominado "
          "por el frente: riesgo de junta fría y la hipótesis de nivel horizontal "
          "no aplica en ese tramo.")

print(f"\n[RESULTADO] t_total = {tiempo_total_final:.3f} s "
      f"(estacionario {mejor['t_est']:.3f} + transitorio {mejor['t_trans']:.3f})")
print(f"            K_fricción = {mejor['K_fric']:.4f} | K_z_efectivo = {K_opt:.4f}")

V_modelo = float(trapezoid(perf["areas_z"], perf["z_vals"]))
masa_modelo = V_modelo * DENSIDAD
caudal_masico = masa_modelo / tiempo_total_final
print(f"[*] Volumen integrado = {V_modelo:.6f} m3 | masa = {masa_modelo:.1f} kg "
      f"| caudal = {caudal_masico:.1f} kg/s")

if V_CAD > 0:
    dif = 100 * (V_modelo - V_CAD) / V_CAD
    print(f"[*] Volumen: modelo {V_modelo:.6f} vs CAD {V_CAD:.6f} ({dif:+.2f} %)")
    if abs(dif) > 1.0:
        raise ValueError(
            f"[GEOMETRÍA] El volumen integrado se aparta {dif:+.2f} % del CAD. "
            "Revisa teselado, fusión de sólidos y cuerpos incluidos antes de "
            "dar por bueno el tiempo.")

print("=" * 74)
print("DIAGNÓSTICO HIDRÁULICO — validez de las hipótesis del modelo")
print("=" * 74)
alertas = []
 
# ---------------------------------------------------------------- 1. CHOKE --
# El modelo usa Q = A_choke*sqrt(2gH/(1+K)). Solo vale si el bebedero es la
# sección mínima y corre lleno. Si los ataques estrangulan más que el
# bebedero, A_choke está mal y todo el caudal está sobreestimado.
ratio_ataques = A_ataques / A_choke
print(f"\n[1] CONTROL DEL CAUDAL")
print(f"    A_choke (bebedero)  = {A_choke:.5f} m2")
print(f"    A_ataques           = {A_ataques:.5f} m2   (ratio {ratio_ataques:.2f})")
if ratio_ataques < 1.0:
    alertas.append("Los ataques estrangulan más que el bebedero: A_choke está mal elegido.")
    print("    [X] Los ATAQUES son el estrangulador real, no el bebedero.")
    print("        Vuelve a correr con d_inf equivalente al área de ataques.")
elif ratio_ataques < 1.2:
    print("    [!] Sistema casi presurizado (ratio < 1.2): el reparto entre "
          "bebedero y ataques es sensible. Verifica que el bebedero corra lleno.")
else:
    print("    [OK] El bebedero es el estrangulador. Sistema no presurizado.")
 
# ------------------------------------------------------- 2. VELOCIDAD Y Fr --
# Q se evalúa al inicio (carga máxima, nivel en z_ataque) y al final
# (nivel en la boca). El Froude en el ataque dice si la vena entra tranquila:
# Fr > 1 = flujo supercrítico, superficie no plana, salto hidráulico.
# La hipótesis de superficie libre horizontal exige Fr < 1.
h_ataque_ef = perf.get("h_ataque_usada", h_ataque)
G_LOC = 9.81
 
def _caudal(nivel_z):
    carga = max(H_total - max(nivel_z, z_ataque), 1e-6)
    return A_choke * np.sqrt(2 * G_LOC * carga / (1.0 + K_opt))
 
Q_ini, Q_fin = _caudal(z_ataque), _caudal(perf["z_vals"][-1])
v_ini, v_fin = Q_ini / A_ataques, Q_fin / A_ataques
Fr_ini = v_ini / np.sqrt(G_LOC * h_ataque_ef)
Fr_fin = v_fin / np.sqrt(G_LOC * h_ataque_ef)
 
print(f"\n[2] VELOCIDAD DE ENTRADA Y RÉGIMEN")
print(f"    Q inicial = {Q_ini*1000:6.2f} L/s | v_ataque = {v_ini:.3f} m/s | Fr = {Fr_ini:.2f}")
print(f"    Q final   = {Q_fin*1000:6.2f} L/s | v_ataque = {v_fin:.3f} m/s | Fr = {Fr_fin:.2f}")
if v_ini > V_CRITICA:
    alertas.append(f"v_ataque inicial {v_ini:.2f} m/s supera la crítica "
                   f"{V_CRITICA} m/s ({v_ini/V_CRITICA:.1f}x): riesgo de arrastre de óxidos.")
    print(f"    [!] Supera la velocidad crítica en {v_ini/V_CRITICA:.1f}x.")
else:
    print(f"    [OK] Por debajo de la velocidad crítica.")
if Fr_ini > 1.0:
    alertas.append(f"Froude inicial {Fr_ini:.2f} > 1: entrada supercrítica, "
                   "la superficie libre NO es plana junto a los ataques.")
    print(f"    [!] Régimen SUPERCRÍTICO al inicio: salto hidráulico en la entrada.")
else:
    print(f"    [OK] Régimen subcrítico: la hipótesis de superficie plana se sostiene.")
 
# ------------------------------------------------ 3. PIEZA VS MAZAROTAS -----
# El tiempo total incluye llenar mazarotas. Para juicios de junta fría lo
# relevante es el tiempo hasta el tope de la PIEZA, no el total.
z_c, t_c = perf["z_vals"], mejor["t_curva"]
t_pieza = float(np.interp(min(z_tope_pieza, z_c[-1]), z_c, t_c))
frac = 100 * t_pieza / mejor["t_total"]
print(f"\n[3] REPARTO PIEZA / MAZAROTAS")
print(f"    t hasta z={z_tope_pieza:.3f} m (tope de pieza) = {t_pieza:6.2f} s ({frac:.1f} %)")
print(f"    t llenando solo mazarotas               = {mejor['t_total']-t_pieza:6.2f} s "
      f"({100-frac:.1f} %)")
print(f"    -> Para riesgo de junta fría usa {t_pieza:.2f} s, no {mejor['t_total']:.2f} s.")
 
# ------------------------------------------------------- 4. FRENTE VS NIVEL -
print(f"\n[4] HIPÓTESIS DE NIVEL HORIZONTAL")
if diag_barrido.get("frente_domina_inicio"):
    tb = diag_barrido["t_barrido_piso_s"]
    alertas.append(f"El frente domina los primeros {tb:.1f} s: nivel no horizontal en ese tramo.")
    print(f"    [!] El frente tarda {tb:.2f} s en recorrer el molde; el nivel supera")
    print(f"        h_ataque antes. Los primeros {tb:.1f} s ({100*tb/mejor['t_total']:.0f} % del "
          "llenado) no cumplen la hipótesis.")
    print(f"        Nivel al terminar el barrido: z = "
          f"{diag_barrido['z_nivel_al_fin_barrido_m']:.4f} m")
else:
    print("    [OK] El nivel sube más despacio que el avance del frente.")
 
# --------------------------------------------------------- 5. CONVERGENCIA --
print(f"\n[5] CALIDAD NUMÉRICA")
print(f"    Malla óptima N = {n_actual} | error {hist_e[-1]:.4f} % | "
      f"convergió: {convergido}")
print(f"    Cortes fallidos: Z={perf['fallos_z']}  X={perf['fallos_x']}")
if len(hist_k) > 2:
    deriva = 100 * abs(hist_k[-1] - hist_k[-2]) / max(hist_k[-1], 1e-9)
    print(f"    Deriva de K_fricción entre las dos últimas mallas: {deriva:.2f} %")
    if deriva > 1.0:
        alertas.append(f"K_fricción aún deriva {deriva:.1f} % entre mallas: "
                       "no está convergido para calibrar.")
        print("    [!] K_fricción no ha convergido. El criterio de parada mira t_total,")
        print("        que es menos sensible. Sube N_MAXIMO si vas a calibrar con esto.")
 
# ------------------------------------------------------------- RESUMEN ------
print("\n" + "=" * 74)
if alertas:
    print(f"{len(alertas)} ADVERTENCIA(S):")
    for i, a in enumerate(alertas, 1):
        print(f"  {i}. {a}")
    print("\nEl tiempo calculado sigue siendo válido como balance de volumen,")
    print("pero el modelo no puede pronunciarse sobre la CALIDAD del llenado.")
else:
    print("Sin advertencias: las hipótesis del modelo se sostienen.")
print("=" * 74)
 
# COMMAND ----------
 
# MAGIC %md
# MAGIC ### Barrido de diseño de ataques
# MAGIC
# MAGIC Recalcula `K_fricción` desde la geometría real para cada `h_ataque`
# MAGIC candidato, así que el acoplamiento (más calado -> más `D_h` -> menos
# MAGIC fricción -> más caudal) queda capturado.
 
# COMMAND ----------
 
def barrido_ataques(alturas, anchos_totales=None):
    """Tabla de diseño: para cada (h, W) devuelve área, régimen y tiempo."""
    filas = []
    for ha in alturas:
        perf_h = extraer_perfiles(STL_PATH, bounds, n_actual, ha,
                                  reparar=AUTO_REPARAR, savgol=FILTRO_SAVGOL,
                                  con_fase_x=True)
        Kf_h, _ = k_friccion(perf_h, f_arena, ha, dinamico=DH_DINAMICO)
        r_h = tiempos_llenado(perf_h, H_total, z_ataque, A_choke,
                              K_sistema, Kf_h, metodo)
        Kze = r_h["K_z_ef"]
        Q = A_choke * np.sqrt(2 * G_LOC * (H_total - z_ataque) / (1.0 + Kze))
        for W in (anchos_totales or [A_ataques / h_ataque_ef]):
            Ag = W * ha
            v = Q / Ag
            filas.append(dict(h_ataque_m=ha, ancho_total_m=W, area_ataques_m2=Ag,
                              ratio_vs_choke=Ag / A_choke, K_friccion=Kf_h,
                              K_z_efectivo=Kze, caudal_Ls=Q * 1000,
                              v_ataque_ms=v,
                              Froude=v / np.sqrt(G_LOC * ha),
                              t_total_s=r_h["t_total"]))
    return pd.DataFrame(filas)
 
# Ejemplo — descomenta para ejecutarlo (una extracción de malla por altura):
# tabla = barrido_ataques([0.018, 0.025, 0.035], anchos_totales=[0.404, 0.606])
# display(tabla)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auditoría y persistencia (esquema Silver)

# COMMAND ----------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.plot(hist_n, hist_t, marker="o", color="navy", linewidth=2)
ax1.set_ylabel("Tiempo teórico (s)")
ax1.set_title(f"Convergencia de malla: {stl_name} ({metodo})")
ax1.grid(True, linestyle="--")
ax2.plot(hist_n[1:], hist_e[1:], marker="s", color="firebrick", linewidth=2)
ax2.axhline(y=TOLERANCIA, color="green", linestyle=":", label=f"Tolerancia {TOLERANCIA}%")
ax2.set_xlabel("Resolución (nodos)"); ax2.set_ylabel("Error relativo (%)")
ax2.set_yscale("log"); ax2.legend(); ax2.grid(True, linestyle="--")

ruta_grafico = f"/Volumes/funcal/silver/resultados_graficos/convergencia_{nombre_limpio}.png"
try:
    plt.savefig(ruta_grafico, dpi=300, bbox_inches="tight")
    print(f"[*] Gráfico guardado en: {ruta_grafico}")
except Exception as e:
    print(f"[!] No se pudo guardar el gráfico: {e}")
plt.close(fig)

# ---- A. Tabla transversal de la integral -----------------------------------
df_integral = pd.DataFrame({
    "altura_z_m": z_opt,
    "area_cavidad_m2": A_opt,
    "carga_efectiva_m": denom_opt ** 2,
    "raiz_carga_efectiva": denom_opt,
    "integrando_fz": fz_opt,
    "tiempo_acumulado_s": mejor["t_curva"],
})
tabla_integral = f"funcal.silver.integral_optima_{nombre_limpio}"
(spark.createDataFrame(df_integral).write.format("delta")
 .mode("overwrite").option("overwriteSchema", "true").saveAsTable(tabla_integral))
print(f"[*] Integral guardada en: {tabla_integral}")

# ---- B. Historial maestro --------------------------------------------------
# pandas -> Spark en vez de createDataFrame([dict]) (inferencia por dict está
# deprecada y ordena las claves alfabéticamente).
registro = pd.DataFrame([{
    "pieza_stl": stl_name,
    "fecha_simulacion": datetime.datetime.now(datetime.timezone.utc),
    "sistema_vaciado": sistema,
    "metodo_numerico": metodo,
    "carga_inicial_hc_m": float(Hc),
    "diametro_inferior_m": float(d_inf),
    "zataque_m": float(z_ataque),
    "h_caida_m": float(h_caida),
    "k_sistema_asumido": float(K_sistema),
    "origen_k_sistema": origen_k,
    "n_inicial": int(N_INICIAL),
    "tolerancia_pct": float(TOLERANCIA),
    "z_boca_m": float(Z_boca),
    "z_base_m": float(Z_base),
    "achoke_m2": float(A_choke),
    "modelo_hibrido_activo": bool(MODELO_HIBRIDO),
    "dh_dinamico_activo": bool(DH_DINAMICO),
    "filtro_savgol_activo": bool(FILTRO_SAVGOL),
    "f_arena": float(f_arena),
    "dh_referencia_m": float(4 * h_ataque),
    "h_ataque_m": float(h_ataque),
    "longitud_molde_m": float(L_molde),
    "k_friccion": float(mejor["K_fric"]),
    "k_efectivo_con_friccion": float(K_opt),
    "carga_total_m": float(H_total),
    "convergio": bool(convergido),
    "error_convergencia_pct": float(hist_e[-1]),
    "diametro_superior_teorico_m": float(d_sup),
    "malla_n_optima": int(n_actual),
    "cortes_fallidos_z": int(perf["fallos_z"]),
    "cortes_fallidos_x": int(perf["fallos_x"]),

    "volumen_ola_m3": float(diag_barrido.get("volumen_ola_m3", 0.0)),
    "z_nivel_al_fin_barrido_m": float(diag_barrido.get("z_nivel_al_fin_barrido_m", 0.0)),
    "frente_domina_inicio": bool(diag_barrido.get("frente_domina_inicio", False)),

    "volumen_integrado_m3": float(V_modelo),
    "densidad_kg_m3": float(DENSIDAD),
    "masa_calculada_kg": float(masa_modelo),
    "caudal_masico_kg_s": float(caudal_masico),

    "t_barrido_piso_diagnostico_s": float(t_horiz),   # NO se suma al total
    "tiempo_fase_1_estacionario_s": float(mejor["t_est"]),
    "tiempo_fase_2_transitorio_s": float(mejor["t_trans"]),
    "tiempo_total_llenado_s": float(tiempo_total_final),

    "eje_barrido": eje_efectivo,
    "eje_barrido_origen": EJE_BARRIDO,
}])

tabla_resumen = "funcal.silver.historial_simulaciones"
(spark.createDataFrame(registro).write.format("delta")
 .mode("append").option("mergeSchema", "true").saveAsTable(tabla_resumen))
print(f"[*] Resultados añadidos a: {tabla_resumen}")