"""
==========================================================================
 SIMULACIÓN DE UN REACTOR PFR DE REFORMADO CON VAPOR (SMR)
==========================================================================
 Modelo de reactor tubular de flujo pistón (PFR) no isotérmico con
 intercambio de calor externo.

 Reacciones:
   R1: CH₄ + H₂O  ⇌  CO + 3 H₂     (Reformado — endotérmica)
   R2: CO  + H₂O  ⇌  CO₂ + H₂      (Water-Gas Shift — exotérmica)

 Se resuelven los balances de materia y energía a lo largo de la masa
 de catalizador (W) mediante integración numérica de EDOs.

 ► Todas las constantes están agrupadas al inicio para facilitar su
   modificación.
==========================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =====================================================================
#  PARÁMETROS MODIFICABLES  ← Cambiar estos valores para experimentar
# =====================================================================
# --- Flujos molares de entrada [kmol/h]
F_CH4_0_kmol_h = 60.0      # Metano
F_H2O_0_kmol_h = 200.0     # Vapor de agua  (ratio S/C = 2)
F_CO_0_kmol_h  = 0.0       # Monóxido de carbono
F_H2_0_kmol_h  = 0.0       # Hidrógeno (traza)
F_CO2_0_kmol_h = 40.0      # Dióxido de carbono
F_N2_0_kmol_h  = 0.0       # Inerte (nitrógeno)

# --- Flujos molares de entrada [mol/s] ---
F_CH4_0 = F_CH4_0_kmol_h * 1000 / 3600   # Metano
F_H2O_0 = F_H2O_0_kmol_h * 1000 / 3600   # Vapor de agua  (ratio S/C = 3)
F_CO_0  = F_CO_0_kmol_h * 1000 / 3600    # Monóxido de carbono
F_H2_0  = F_H2_0_kmol_h * 1000 / 3600    # Hidrógeno (traza)
F_CO2_0 = F_CO2_0_kmol_h * 1000 / 3600   # Dióxido de carbono
F_N2_0  = F_N2_0_kmol_h * 1000 / 3600    # Inerte (nitrógeno)

# --- Condiciones de operación ---
T0      = 800 + 273.15   # Temperatura de entrada [K]  (800 °C)
P_total = 20.0           # Presión total [atm]

# --- Masa de catalizador ---
W_total = 20           # Masa total de catalizador [kg]

# --- Parámetros cinéticos  (k = A · exp(-Ea / R·T)) ---
#     Cinética tipo ley de potencias con fuerza impulsora de equilibrio
#
# Reacción 1:  CH₄ + H₂O → CO + 3 H₂
A1  = 5.0e8              # Factor pre-exponencial  [mol / (kg_cat · s · atm²)]
Ea1 = 240.1e3            # Energía de activación    [J/mol]
#
# Reacción 2:  CO + H₂O → CO₂ + H₂
A2  = 2.5e2              # Factor pre-exponencial  [mol / (kg_cat · s · atm²)]
Ea2 = 67.13e3            # Energía de activación    [J/mol]

# --- Constantes de equilibrio (ln Kp = a − b/T) ---
#   R1: Kp₁ [atm²]    →  valores típicos del reformado con vapor
Kp1_a = 30.420           # Parámetro adimensional
Kp1_b = 27_106.0         # [K]
#   R2: Kp₂ [adim.]   →  Water-Gas Shift
Kp2_a = -3.798
Kp2_b = 4_160.0          # [K]

# --- Entalpías de reacción [J/mol] (a las condiciones de referencia) ---
DH_R1 =  206.1e3         # ΔH₁  (endotérmica, positiva)
DH_R2 = -41.2e3         # ΔH₂  (exotérmica, negativa)

# --- Capacidades caloríficas promedio [J/(mol·K)] ---
Cp = {
    "CH4": 76.5,
    "H2O": 41.5,
    "CO":  30.5,
    "H2":  29.0,
    "CO2": 45.0,
    "N2":  30.0,
}

# --- Transferencia de calor con el exterior (Horno de reformado) ---
# Modelo basado en las dimensiones físicas del tubo y propiedades del lecho

D_tubo  = 0.1          # Diámetro interno del tubo de reformado [m] (Típico: 0.1 - 0.15 m)
rho_bed = 1000.0       # Densidad del lecho catalítico (Ni/Al2O3) [kg/m³]
U_coef  = 300.0        # Coeficiente global de transferencia térmica U [J/(s·m²·K)]

# Cálculo del área de transferencia específica por masa de catalizador [m²/kg_cat]
# a = Área_lateral / Masa_catalizador = 4 / (D * rho_bed)
area_esp = 4.0 / (D_tubo * rho_bed)

# Ua: Coeficiente global volumétrico [J/(s·kg_cat·K)]
Ua = U_coef * area_esp  # Resulta en 12.0 bajo estos parámetros geométricos

# Temperatura de la pared externa de los tubos (Radiación del quemador)
# Debe ser sustancialmente superior a los 750-850 °C requeridos por el gas de reacción
T_ext = 1000.0 + 273.15 # Temperatura del medio exterior [K] (Elevada a 1000 °C para mayor rigor industrial)

# --- Constante universal de gases ---
R_GAS = 8.314            # [J/(mol·K)]


# =====================================================================
#  MODELO DEL REACTOR  (no modificar salvo que se conozca el modelo)
# =====================================================================

def Kp1(T):
    """Constante de equilibrio de R1 [atm²]."""
    return np.exp(Kp1_a - Kp1_b / T)

def Kp2(T):
    """Constante de equilibrio de R2 [adimensional]."""
    return np.exp(Kp2_a + Kp2_b / T)

def tasas_reaccion(T, p_CH4, p_H2O, p_CO, p_H2, p_CO2):
    """
    Calcula las velocidades de reacción r₁ y r₂ [mol/(kg_cat·s)].
    Incluye término de fuerza impulsora hacia el equilibrio.
    """
    k1 = A1 * np.exp(-Ea1 / (R_GAS * T))
    k2 = A2 * np.exp(-Ea2 / (R_GAS * T))

    # Fuerza impulsora: (productos/reactivos) comparado con equilibrio
    # R1: CH₄ + H₂O ⇌ CO + 3H₂   →  Kp₁ = p_CO · p_H₂³ / (p_CH₄ · p_H₂O)
    Kp1_val = Kp1(T)
    driving1 = p_CH4 * p_H2O - (p_CO * p_H2**3) / Kp1_val

    # R2: CO + H₂O ⇌ CO₂ + H₂    →  Kp₂ = p_CO₂ · p_H₂ / (p_CO · p_H₂O)
    Kp2_val = Kp2(T)
    driving2 = p_CO * p_H2O - (p_CO2 * p_H2) / Kp2_val

    r1 = k1 * max(driving1, 0.0)   # evitar tasas negativas netas artificiales
    r2 = k2 * max(driving2, 0.0)

    return r1, r2


def reactor_odes(W, y):
    """
    Sistema de EDOs:  dy/dW  donde W = masa de catalizador [kg].
    y = [F_CH4, F_H2O, F_CO, F_H2, F_CO2, T]
    """
    F_CH4, F_H2O, F_CO, F_H2, F_CO2, T = y

    # Evitar flujos negativos (protección numérica)
    F_CH4 = max(F_CH4, 1e-12)
    F_H2O = max(F_H2O, 1e-12)
    F_CO  = max(F_CO,  1e-12)
    F_H2  = max(F_H2,  1e-12)
    F_CO2 = max(F_CO2, 1e-12)

    # Flujo total
    F_total = F_CH4 + F_H2O + F_CO + F_H2 + F_CO2 + F_N2_0

    # Presiones parciales [atm]
    p_CH4 = (F_CH4 / F_total) * P_total
    p_H2O = (F_H2O / F_total) * P_total
    p_CO  = (F_CO  / F_total) * P_total
    p_H2  = (F_H2  / F_total) * P_total
    p_CO2 = (F_CO2 / F_total) * P_total

    # Velocidades de reacción
    r1, r2 = tasas_reaccion(T, p_CH4, p_H2O, p_CO, p_H2, p_CO2)

    # Balances molares  dFi/dW
    dF_CH4 = -r1
    dF_H2O = -r1 - r2
    dF_CO  =  r1 - r2
    dF_H2  =  3.0 * r1 + r2
    dF_CO2 =  r2

    # Balance de energía  dT/dW
    sum_FiCpi = (F_CH4 * Cp["CH4"] + F_H2O * Cp["H2O"] +
                 F_CO  * Cp["CO"]  + F_H2  * Cp["H2"]  +
                 F_CO2 * Cp["CO2"] + F_N2_0 * Cp["N2"])

    Q_ext    = Ua * (T_ext - T)          # calor del exterior
    Q_rxn    = -r1 * DH_R1 - r2 * DH_R2  # calor neto de reacciones
    dT       = (Q_ext + Q_rxn) / sum_FiCpi

    return [dF_CH4, dF_H2O, dF_CO, dF_H2, dF_CO2, dT]


# =====================================================================
#  RESOLUCIÓN DEL SISTEMA DE EDOS
# =====================================================================

y0 = [F_CH4_0, F_H2O_0, F_CO_0, F_H2_0, F_CO2_0, T0]
W_span = (0, W_total)
W_eval = np.linspace(0, W_total, 2000)

sol = solve_ivp(
    reactor_odes,
    W_span,
    y0,
    method="LSODA",        # Método robusto (cambia auto entre stiff/non-stiff)
    t_eval=W_eval,
    rtol=1e-6,
    atol=1e-8,
)

if not sol.success:
    print(f"AVISO: La integracion no convergio: {sol.message}")
else:
    print("OK - Integracion completada con exito.")

# Extraer resultados
W      = sol.t
F_CH4  = sol.y[0]
F_H2O  = sol.y[1]
F_CO   = sol.y[2]
F_H2   = sol.y[3]
F_CO2  = sol.y[4]
T      = sol.y[5]

# Conversión de metano:  X = (F_CH4,0 − F_CH4) / F_CH4,0
X_CH4 = (F_CH4_0 - F_CH4) / F_CH4_0

# Temperatura en °C
T_celsius = T - 273.15


# =====================================================================
#  GRÁFICOS
# =====================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Simulación del Reactor de Reformado con Vapor (SMR)\n"
    r"R1: CH$_4$ + H$_2$O $\rightleftharpoons$ CO + 3 H$_2$   |   "
    r"R2: CO + H$_2$O $\rightleftharpoons$ CO$_2$ + H$_2$",
    fontsize=13, fontweight="bold",
)

# --- 1) Temperatura y Conversión vs. Masa de catalizador ---
ax1 = axes[0, 0]
color_T = "#d62728"
color_X = "#1f77b4"

ax1.set_xlabel("Masa de catalizador, W [kg]", fontsize=11)
ax1.set_ylabel("Temperatura [°C]", color=color_T, fontsize=11)
ln1 = ax1.plot(W, T_celsius, color=color_T, linewidth=2, label="T [°C]")
ax1.tick_params(axis="y", labelcolor=color_T)
ax1.set_ylim(
    min(T_celsius) - 20,
    max(T_celsius) + 20,
)

ax1b = ax1.twinx()
ax1b.set_ylabel("Conversión CH₄ (X)", color=color_X, fontsize=11)
ln2 = ax1b.plot(W, X_CH4, color=color_X, linewidth=2, linestyle="--", label="X CH₄")
ax1b.tick_params(axis="y", labelcolor=color_X)
ax1b.set_ylim(-0.02, 1.02)

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right", fontsize=10)
ax1.set_title("Temperatura y Conversión vs. W", fontsize=11)
ax1.grid(True, alpha=0.3)

# --- 2) Conversión vs. Temperatura (gráfico paramétrico) ---
ax2 = axes[0, 1]
ax2.plot(T_celsius, X_CH4, color="#2ca02c", linewidth=2.5)
ax2.set_xlabel("Temperatura [°C]", fontsize=11)
ax2.set_ylabel("Conversión CH₄ (X)", fontsize=11)
ax2.set_title("Conversión vs. Temperatura", fontsize=11)
ax2.set_ylim(-0.02, 1.02)
ax2.grid(True, alpha=0.3)
# Marcar inicio y fin
ax2.annotate(
    "Entrada",
    xy=(T_celsius[0], X_CH4[0]),
    fontsize=9, fontweight="bold",
    xytext=(15, 15), textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="gray"),
)
ax2.annotate(
    "Salida",
    xy=(T_celsius[-1], X_CH4[-1]),
    fontsize=9, fontweight="bold",
    xytext=(-50, -20), textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="gray"),
)

# --- 3) Flujos molares vs. W ---
ax3 = axes[1, 0]
ax3.plot(W, F_CH4, label=r"CH$_4$", linewidth=2)
ax3.plot(W, F_H2O, label=r"H$_2$O", linewidth=2)
ax3.plot(W, F_CO,  label="CO",       linewidth=2)
ax3.plot(W, F_H2,  label=r"H$_2$",   linewidth=2)
ax3.plot(W, F_CO2, label=r"CO$_2$",  linewidth=2)
ax3.set_xlabel("Masa de catalizador, W [kg]", fontsize=11)
ax3.set_ylabel("Flujo molar [mol/s]", fontsize=11)
ax3.set_title("Flujos molares vs. W", fontsize=11)
ax3.legend(fontsize=10, ncol=2)
ax3.grid(True, alpha=0.3)

# --- 4) Composición molar a la salida (diagrama de barras) ---
ax4 = axes[1, 1]
especies = [r"CH$_4$", r"H$_2$O", "CO", r"H$_2$", r"CO$_2$", r"N$_2$"]
flujos_salida = [F_CH4[-1], F_H2O[-1], F_CO[-1], F_H2[-1], F_CO2[-1], F_N2_0]
F_total_sal = sum(flujos_salida)
fracciones = [f / F_total_sal * 100 for f in flujos_salida]
colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

bars = ax4.bar(especies, fracciones, color=colores, edgecolor="black", linewidth=0.5)
ax4.set_ylabel("Fracción molar [%]", fontsize=11)
ax4.set_title("Composición a la salida del reactor", fontsize=11)
ax4.grid(True, axis="y", alpha=0.3)
for bar, frac in zip(bars, fracciones):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{frac:.1f}%",
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("resultado_reactor_SMR.png", dpi=150, bbox_inches="tight")
plt.show()


# =====================================================================
#  RESUMEN DE RESULTADOS
# =====================================================================
print("\n" + "=" * 60)
print("  RESUMEN DE RESULTADOS DEL REACTOR SMR")
print("=" * 60)
print(f"  Temperatura entrada:   {T0 - 273.15:.1f} C")
print(f"  Temperatura salida:    {T_celsius[-1]:.1f} C")
print(f"  Conversion CH4:       {X_CH4[-1] * 100:.2f} %")
print(f"  Presion:               {P_total} atm")
print(f"  Masa de catalizador:   {W_total} kg")
print(f"  Ratio S/C:             {F_H2O_0 / F_CH4_0:.1f}")
print("-" * 60)
print("  Flujos molares a la salida [mol/s]:")
print(f"    CH4  = {F_CH4[-1]:8.4f}")
print(f"    H2O  = {F_H2O[-1]:8.4f}")
print(f"    CO   = {F_CO[-1]:8.4f}")
print(f"    H2   = {F_H2[-1]:8.4f}")
print(f"    CO2  = {F_CO2[-1]:8.4f}")
print(f"    N2   = {F_N2_0:8.4f}")
print(f"    TOTAL= {F_total_sal:8.4f}")
print("=" * 60)
