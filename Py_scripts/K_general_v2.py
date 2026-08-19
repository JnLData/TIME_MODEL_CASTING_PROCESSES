# Databricks notebook source
# MAGIC %md
# MAGIC # Calibración inversa de K_sistema (v2)
# MAGIC Requiere `_core_llenado` en la misma carpeta. Usa exactamente la misma
# MAGIC geometría y física que `Tiempo_Llenado_Colada_General_v2`.

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
from scipy.optimize import minimize_scalar

# ============================================================================
# 1. WIDGETS  (deben coincidir con los de la simulación)
# ============================================================================
dbutils.widgets.text("1_archivo_stl", "711-021-A_REV00_SIM02_ENSAMBLE.STL", "1. Archivo STL")
dbutils.widgets.dropdown("2_sistema", "Copa", ["Balsa", "Copa"], "2. Sistema")
dbutils.widgets.text("3_Hc", "0.666", "3. Carga Hc (m)")
dbutils.widgets.text("4_h_caida", "0.250", "4. Caída cuchara (m)")
dbutils.widgets.text("5_d_inf", "0.070", "5. Diámetro choke (m)")
dbutils.widgets.text("6_zataque", "0.046456",
                     "6. Cota Z del EJE de ataques (m sobre el fondo de la pieza)")
dbutils.widgets.text("7_h_ataque", "0.018", "7. Altura física orificio ataque (m)")
dbutils.widgets.text("8_f_arena", "0.045", "8. Factor fricción arena")
dbutils.widgets.text("9_Malla_N", "1600", "9. Malla (N) — usar la N convergida")
dbutils.widgets.dropdown("10_Modo_Calibracion", "Simple_1_Punto",
                         ["Simple_1_Punto", "Curva_Bronce_SSE"], "10. Modo")
dbutils.widgets.text("11_T_planta", "52", "11. Tiempo total medido (s)")
dbutils.widgets.text("12_Vol_canales", "0.005982078", "12. Vol. bebedero+canales (m3)")
dbutils.widgets.text("13_Tabla_Bronce", "funcal.bronce.alturas_tiempos_planta", "13. Tabla mediciones")
dbutils.widgets.dropdown("14_autoreparar", "No", ["No", "Si"], "14. Auto-Reparar STL")
dbutils.widgets.dropdown("15_metodo", "Simpson", ["Simpson", "Trapezoidal"], "15. Método numérico")
dbutils.widgets.dropdown("16_filtro_savgol", "No", ["No", "Si"], "16. Filtro Savitzky-Golay")
dbutils.widgets.dropdown("17_Dh_dinamico", "Si", ["No", "Si"], "17. D_h Dinámico [Opción B]")
dbutils.widgets.text("18_densidad", "8800", "18. Densidad de la aleación (kg/m3)")
dbutils.widgets.text("19_volumen_cad", "0", "19. Volumen del CAD (m3, 0 = no comprobar)")

dbutils.widgets.dropdown("20_eje_barrido", "Auto", ["Auto", "X", "Y"],
                         "20. Eje de avance del frente")

dbutils.widgets.dropdown("21_modelo_hibrido", "Si", ["No", "Si"], "21. Activar Fase X (Híbrido)")

stl_name = dbutils.widgets.get("1_archivo_stl").strip()
STL_PATH = f"/Volumes/funcal/bronce/input_data/{stl_name}"
sistema = dbutils.widgets.get("2_sistema")
Hc = float(dbutils.widgets.get("3_Hc"))
h_caida = float(dbutils.widgets.get("4_h_caida")) if sistema == "Copa" else 0.0
H_total = Hc + h_caida
d_inf = float(dbutils.widgets.get("5_d_inf"))
A_choke = np.pi * d_inf ** 2 / 4
z_ataque = float(dbutils.widgets.get("6_zataque"))
h_ataque = float(dbutils.widgets.get("7_h_ataque"))
f_arena = float(dbutils.widgets.get("8_f_arena"))
num_cortes = int(dbutils.widgets.get("9_Malla_N"))
MODO = dbutils.widgets.get("10_Modo_Calibracion")
t_planta = float(dbutils.widgets.get("11_T_planta"))
vol_canales = float(dbutils.widgets.get("12_Vol_canales"))
tabla_bronce = dbutils.widgets.get("13_Tabla_Bronce")
AUTO_REPARAR = dbutils.widgets.get("14_autoreparar") == "Si"
metodo = dbutils.widgets.get("15_metodo")
FILTRO_SAVGOL = dbutils.widgets.get("16_filtro_savgol") == "Si"
DH_DINAMICO = dbutils.widgets.get("17_Dh_dinamico") == "Si"
DENSIDAD = float(dbutils.widgets.get("18_densidad"))
V_CAD = float(dbutils.widgets.get("19_volumen_cad"))   # 0 = sin comprobar
EJE_BARRIDO = dbutils.widgets.get("20_eje_barrido")
MODELO_HIBRIDO = dbutils.widgets.get("21_modelo_hibrido") == "Si"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extracción geométrica (una sola pasada — no depende de K)

# COMMAND ----------

bounds, diag = cargar_malla_driver(STL_PATH, AUTO_REPARAR)
print(f"[*] {stl_name} | extents {np.round(diag['extents_m'],4)} m | "
      f"cuerpos {diag['n_cuerpos']} | watertight {diag['watertight']}")

# Mismo criterio que la simulación: si difieren, el K calibrado no es válido allí.
lx = bounds[1, 0] - bounds[0, 0]
ly = bounds[1, 1] - bounds[0, 1]
if EJE_BARRIDO in ("X", "Y"):
    eje_efectivo = EJE_BARRIDO
else:
    eje_efectivo = "X" if lx >= ly else "Y"
    if max(lx, ly) > 0 and abs(lx - ly) / max(lx, ly) < 0.05:
        print(f"    [!] Lx={lx:.4f} y Ly={ly:.4f} difieren menos del 5 %: la "
              "elección automática de eje es frágil. Fíjalo con el widget 20.")

L_barrido = lx if eje_efectivo == "X" else ly
K_fric_estimado = f_arena * L_barrido / (4 * h_ataque)
CON_FASE_X = MODELO_HIBRIDO and K_fric_estimado > 0.15

print(f"[*] Eje de barrido: {eje_efectivo} (L = {L_barrido:.3f} m, origen: {EJE_BARRIDO})")
print(f"[*] K_fricción estimado: {K_fric_estimado:.3f} | "
      f"Fase X: {'activa' if CON_FASE_X else 'DESACTIVADA'}")

perfiles = extraer_perfiles(STL_PATH, bounds, num_cortes, h_ataque,
                            reparar=AUTO_REPARAR, savgol=FILTRO_SAVGOL,
                            con_fase_x=CON_FASE_X,
                            eje_forzado=None if EJE_BARRIDO == "Auto" else EJE_BARRIDO)
K_fric, _ = k_friccion(perfiles, f_arena, h_ataque, dinamico=DH_DINAMICO)
print(f"[*] K_fricción acumulada = {K_fric:.4f} "
      f"(Dh {'dinámico' if DH_DINAMICO else 'estático'})")
print(f"[*] Cortes fallidos: Z={perfiles['fallos_z']}  X={perfiles['fallos_x']}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Solver inverso — forma cerrada

# COMMAND ----------

print(f"--- CALIBRACIÓN ({MODO}) ---")

if MODO == "Simple_1_Punto":
    # ------------------------------------------------------------------
    # t(K) = base*sqrt(1+K+Kf) + c*sqrt(1+K) es invertible analíticamente.
    # Ni root_scalar, ni bracket [1,30] que pueda no contener la raíz, ni
    # K_calibrado sin definir tras un ValueError capturado.
    # ------------------------------------------------------------------
    cal = calibrar_K(t_planta, perfiles, H_total, z_ataque, A_choke, K_fric,
                     vol_canales, metodo)
    K_calibrado = cal["K_sistema"]
    SSE_minimo = 0.0
    n_puntos = 1

    print(f"[*] Tiempo medido en planta      : {t_planta:.2f} s")
    print(f"[*] Mínimo físico (K_sistema=0)  : {cal['t_min_fisico']:.2f} s")
    print(f"[*] t_cebado (coherente con K)   : {cal['t_cebado']:.3f} s")
    print(f"[*] Integral I                   : {cal['I']:.5f}")
    print(f"[*] K_fricción                   : {K_fric:.4f}")
    print(f"[*] K_sistema (K_ge) CALIBRADO   : {K_calibrado:.4f}")
    print(f"[*] K_z_efectivo                 : {cal['K_z_ef']:.4f}")
    if K_calibrado < 0.2:
        print("    [!] K muy bajo: revisa d_inf, o el sistema tiene menos pérdida de la esperada.")
    if K_calibrado > 40:
        print("    [!] K muy alto: revisa H_total, el volumen de la pieza o el cronometraje.")

elif MODO == "Curva_Bronce_SSE":
    # ------------------------------------------------------------------
    # La curva simulada es t(z,K) = sqrt(1+K+Kf)*G(z), con G precalculada.
    # El optimizador ya NO reintegra la geometría en cada evaluación.
    # ------------------------------------------------------------------
    f_z, _ = integrando_z(perfiles, H_total, z_ataque)
    z_sim = perfiles["z_vals"]
    # curva geométrica: t(z, K) = sqrt(1 + K + K_fric) * curva_G(z)
    curva_G = cumulative_trapezoid(y=f_z, x=z_sim, initial=0.0) / (A_choke * np.sqrt(2 * G))
    c = vol_canales / (A_choke * np.sqrt(2 * G * H_total))

    df_b = spark.read.table(tabla_bronce).toPandas()
    z_real = df_b["altura_medida_m"].values.astype(float)
    t_med = df_b["tiempo_medido_s"].values.astype(float)

    # Interpolación SIN extrapolar: fuera del rango simulado no hay física.
    dentro = (z_real >= z_sim.min()) & (z_real <= z_sim.max())
    if (~dentro).any():
        print(f"    [!] {int((~dentro).sum())} puntos fuera del rango Z simulado: descartados.")
    z_real, t_med = z_real[dentro], t_med[dentro]
    n_puntos = int(len(z_real))
    if n_puntos < 4:
        raise ValueError(
            f"[DATOS] Solo {n_puntos} puntos de control dentro del rango Z. "
            "El modo SSE necesita al menos 4, idealmente repartidos por encima "
            "y por debajo de z_ataque.")
    G_i = np.interp(z_real, z_sim, curva_G)

    def sse(s):                       # s = sqrt(1+K_sistema)
        t_sim = np.sqrt(max(s * s + K_fric, 1e-12)) * G_i + c * s
        return float(np.sum((t_sim - t_med) ** 2))

    LO, HI = 1.0, 12.0                # s en [1,12] => K en [0,143]
    res = minimize_scalar(sse, bounds=(LO, HI), method="bounded")
    s_opt = float(res.x)
    if abs(s_opt - LO) < 1e-3 or abs(s_opt - HI) < 1e-3:
        print("    [!] El óptimo cayó en el BORDE del intervalo: la calibración no es fiable.")
    K_calibrado = s_opt ** 2 - 1.0
    SSE_minimo = float(res.fun)
    
    # Contraste independiente: la de 1 punto ancla solo en el tiempo total.
    # No alimenta la SSE — sirve para detectar desajustes de FORMA de la curva.
    if t_planta > 0:
        try:
            ref = calibrar_K(t_planta, perfiles, H_total, z_ataque, A_choke,
                             K_fric, vol_canales, metodo)
            desv = 100 * (K_calibrado - ref["K_sistema"]) / ref["K_sistema"]
            print(f"[*] Contraste 1 punto (t={t_planta:.1f} s): K = {ref['K_sistema']:.4f} "
                  f"| SSE se desvía {desv:+.1f} %")
            if abs(desv) > 6.0:
                print("    [!] Discrepancia por encima del ruido esperado. El modelo "
                      "reproduce el tiempo TOTAL pero no la FORMA de la curva. "
                      "Sospecha del tramo inicial dominado por el frente.")
        except ValueError as e:
            print(f"    [i] Contraste no disponible: {e}")
            
    print(f"[*] Puntos de control            : {n_puntos}")
    print(f"[*] t_cebado (coherente con K)   : {c*s_opt:.3f} s")
    print(f"[*] Error residual (SSE)         : {SSE_minimo:.4f}")
    print(f"[*] RMSE                         : {np.sqrt(SSE_minimo/max(n_puntos,1)):.3f} s")
    print(f"[*] K_sistema (K_ge) CALIBRADO   : {K_calibrado:.4f}")

V_modelo = float(trapezoid(perfiles["areas_z"], perfiles["z_vals"]))
masa_modelo = V_modelo * DENSIDAD
# El caudal solo tiene sentido contra el tiempo medido, que únicamente
# interviene en la calibración de 1 punto.
caudal_masico = (masa_modelo / t_planta
                 if (MODO == "Simple_1_Punto" and t_planta > 0) else np.nan)
print(f"[*] Volumen integrado = {V_modelo:.6f} m3 | masa = {masa_modelo:.1f} kg"
      + (f" | caudal = {caudal_masico:.1f} kg/s" if np.isfinite(caudal_masico) else ""))

if V_CAD > 0:
    dif = 100 * (V_modelo - V_CAD) / V_CAD
    print(f"[*] Volumen: modelo {V_modelo:.6f} vs CAD {V_CAD:.6f} ({dif:+.2f} %)")
    if abs(dif) > 1.0:
        raise ValueError(
            f"[GEOMETRÍA] El volumen integrado se aparta {dif:+.2f} % del CAD. "
            "Revisa teselado, fusión de sólidos y cuerpos incluidos antes de "
            "dar por bueno el tiempo.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persistencia — cierra el lazo con la simulación

# COMMAND ----------

# La simulación lee esta tabla cuando el widget 8_Ksistema se deja vacío,
# así que ya no hace falta copiar el número a mano.
registro = pd.DataFrame([{
    "pieza_stl": stl_name,
    "fecha_calibracion": datetime.datetime.now(datetime.timezone.utc),
    "modo": MODO,
    "k_sistema": float(K_calibrado),
    "k_friccion": float(K_fric),
    "k_z_efectivo": float(K_calibrado + K_fric),
    "t_planta_s": float(t_planta) if MODO == "Simple_1_Punto" else np.nan,
    "vol_canales_m3": float(vol_canales),
    "h_total_m": float(H_total),
    "achoke_m2": float(A_choke),
    "z_ataque_m": float(z_ataque),
    "h_ataque_m": float(h_ataque),
    "f_arena": float(f_arena),
    "malla_n": int(num_cortes),
    "metodo_numerico": metodo,
    "n_puntos_control": int(n_puntos),
    "sse": float(SSE_minimo),
    "cortes_fallidos_z": int(perfiles["fallos_z"]),
    "cortes_fallidos_x": int(perfiles["fallos_x"]),
    "dh_dinamico_activo": bool(DH_DINAMICO),
    "volumen_integrado_m3": float(V_modelo),
    "densidad_kg_m3": float(DENSIDAD),
    "masa_calculada_kg": float(masa_modelo),
    "caudal_masico_kg_s": float(caudal_masico),
    "modelo_hibrido_activo": bool(CON_FASE_X),
    "eje_barrido": eje_efectivo,
}])

TABLA_CALIB = "funcal.silver.calibracion_k_sistema"
(spark.createDataFrame(registro).write.format("delta")
 .mode("append").option("mergeSchema", "true").saveAsTable(TABLA_CALIB))
print(f"[*] Calibración guardada en: {TABLA_CALIB}")