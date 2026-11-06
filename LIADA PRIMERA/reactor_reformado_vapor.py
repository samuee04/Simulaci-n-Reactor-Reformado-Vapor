"""
==========================================================================
 SIMULACIÓN DE UN REACTOR PFR DE REFORMADO CON VAPOR (SMR)
 Modelo Cinético Intrínseco LHHW de (Xu & Froment, 1989)
==========================================================================
 Modelo de reactor tubular de flujo pistón (PFR) no isotérmico con
 intercambio de calor externo sobre catalizador Ni/MgAl2O4.

 Reacciones (Xu & Froment, 1989):
   R1: CH₄ + H₂O   ⇌  CO  + 3 H₂     (Reformado a CO — endotérmica)
   R2: CO  + H₂O   ⇌  CO₂ + H₂       (Water-Gas Shift — exotérmica)
   R3: CH₄ + 2 H₂O ⇌  CO₂ + 4 H₂     (Reformado a CO₂ — endotérmica)

 Se resuelven los balances de materia y energía a lo largo de la masa
 de catalizador (W) mediante integración numérica de EDOs.
==========================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =====================================================================
#  PARÁMETROS MODIFICABLES  ← Cambiar estos valores para experimentar
# =====================================================================
# --- Flujos molares de entrada [kmol/h] ---
F_CH4_0_kmol_h = 60.0      # Metano
F_H2O_0_kmol_h = 200.0     # Vapor de agua (ratio S/C = 3.33)
F_CO_0_kmol_h  = 0.0       # Monóxido de carbono
# CAMBIO 1: En (Xu & Froment, 1989), p_H2 no puede ser exactamente 0 en la
# entrada porque divide en r1 (p_H2^2.5), r2 (p_H2), r3 (p_H2^3.5) y DEN.
F_H2_0_kmol_h  = 1.0       # Hidrógeno (traza necesaria en alimentación)
F_CO2_0_kmol_h = 40.0      # Dióxido de carbono
F_N2_0_kmol_h  = 0.0       # Inerte (nitrógeno)

# --- Conversión de flujos molares de entrada a [mol/s] ---
F_CH4_0 = F_CH4_0_kmol_h * 1000.0 / 3600.0
F_H2O_0 = F_H2O_0_kmol_h * 1000.0 / 3600.0
F_CO_0  = F_CO_0_kmol_h  * 1000.0 / 3600.0
F_H2_0  = F_H2_0_kmol_h  * 1000.0 / 3600.0
F_CO2_0 = F_CO2_0_kmol_h * 1000.0 / 3600.0
F_N2_0  = F_N2_0_kmol_h  * 1000.0 / 3600.0

# --- Condiciones de operación ---
T0          = 800.0 + 273.15         # Temperatura de entrada [K] (800 °C)
P_total_atm = 20.0                   # Presión total [atm]
# CAMBIO 2: El modelo de (Xu & Froment, 1989) trabaja estrictamente en [bar]
P_total_bar = P_total_atm * 1.01325  # Presión total [bar] (20.265 bar)

# --- Masa y eficacia del catalizador (Ni/MgAl2O4, 15.2% Ni) ---
W_total = 3000.0   # Masa total de catalizador [kg]
# Factor de eficacia catalítica (1.0 = cinética intrínseca pura;
# 0.05 - 0.10 = típico en anillos industriales con difusión interna)
eta_cat = 1.0

# --- Constante universal de gases ---
R_GAS = 8.314      # [J/(mol·K)]

# =====================================================================
#  CAMBIO 3: PARÁMETROS CINÉTICOS Y DE ADSORCIÓN (Xu & Froment, 1989)
# =====================================================================
# --- Constantes cinéticas: k_i = A_i * exp(-Ea_i / (R * T)) ---
# R1: CH₄ + H₂O ⇌ CO + 3 H₂
A1  = 4.225e15     # [kmol · bar^0.5 / (kg_cat · h)]
Ea1 = 240.1e3      # [J/mol]

# R2: CO + H₂O ⇌ CO₂ + H₂
A2  = 1.955e6      # [kmol / (kg_cat · h · bar)]
Ea2 = 67.13e3      # [J/mol]

# R3: CH₄ + 2 H₂O ⇌ CO₂ + 4 H₂
A3  = 1.020e15     # [kmol · bar^0.5 / (kg_cat · h)]
Ea3 = 243.9e3      # [J/mol]

# --- Constantes de adsorción: K_j = A_j * exp(-dH_ads_j / (R * T)) ---
A_CO   = 8.23e-5   # [bar^-1]
dH_CO  = -70.65e3  # Entalpía de adsorción de CO  [J/mol]

A_H2   = 6.12e-9   # [bar^-1]
dH_H2  = -82.90e3  # Entalpía de adsorción de H2  [J/mol]

A_CH4  = 6.65e-4   # [bar^-1]
dH_CH4 = -38.28e3  # Entalpía de adsorción de CH4 [J/mol]

A_H2O  = 1.77e5    # [adimensional]
dH_H2O = 88.68e3   # Entalpía de adsorción disociativa de H2O [J/mol]

# --- Constantes de equilibrio termodinámico ---
# R1: Kp1 originalmente en [atm²], se convierte a [bar²] en la función K1(T)
Kp1_a = 30.420
Kp1_b = 27_106.0   # [K]
# R2: Kp2 [adimensional] (Water-Gas Shift)
Kp2_a = -3.798
Kp2_b = 4_160.0    # [K]

# --- Entalpías de reacción [J/mol] (Xu & Froment, 1989) ---
DH_R1 =  206.1e3   # ΔH₁ (endotérmica)
DH_R2 =  -41.15e3  # ΔH₂ (exotérmica)
DH_R3 =  164.9e3   # ΔH₃ = ΔH₁ + ΔH₂ (endotérmica)

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
D_tubo   = 0.1              # Diámetro interno del tubo de reformado [m]
rho_bed  = 1000.0           # Densidad aparente del lecho (Ni/MgAl2O4) [kg/m³]
U_coef   = 300.0            # Coeficiente global U [J/(s·m²·K)]
area_esp = 4.0 / (D_tubo * rho_bed)  # Área específica [m²/kg_cat]
Ua       = U_coef * area_esp         # Coeficiente volumétrico [J/(s·kg_cat·K)]
T_ext    = 1000.0 + 273.15  # Temperatura del horno exterior [K]


# =====================================================================
#  MODELO DEL REACTOR (Ecuaciones de Xu & Froment, 1989)
# =====================================================================

def K1_eq(T):
    """Constante de equilibrio de R1 en [bar²]."""
    return np.exp(Kp1_a - Kp1_b / T) * (1.01325 ** 2)

def K2_eq(T):
    """Constante de equilibrio de R2 [adimensional]."""
    return np.exp(Kp2_a + Kp2_b / T)

def K3_eq(T):
    """Constante de equilibrio de R3 en [bar²] (K3 = K1 * K2)."""
    return K1_eq(T) * K2_eq(T)

def tasas_reaccion(T, p_CH4, p_H2O, p_CO, p_H2, p_CO2):
    """
    Calcula las velocidades de reacción r₁, r₂ y r₃ en [mol/(kg_cat·s)]
    según el mecanismo LHHW de (Xu & Froment, 1989).
    Las presiones parciales p_j entran en [bar].
    """
    # 1. Constantes de velocidad intrínseca (Arrhenius)
    k1 = A1 * np.exp(-Ea1 / (R_GAS * T))
    k2 = A2 * np.exp(-Ea2 / (R_GAS * T))
    k3 = A3 * np.exp(-Ea3 / (R_GAS * T))

    # 2. Constantes de adsorción (van't Hoff)
    K_CO  = A_CO  * np.exp(-dH_CO  / (R_GAS * T))
    K_H2  = A_H2  * np.exp(-dH_H2  / (R_GAS * T))
    K_CH4 = A_CH4 * np.exp(-dH_CH4 / (R_GAS * T))
    K_H2O = A_H2O * np.exp(-dH_H2O / (R_GAS * T))

    # 3. Constantes de equilibrio
    K1 = K1_eq(T)
    K2 = K2_eq(T)
    K3 = K3_eq(T)

    # 4. Denominador común de adsorción (DEN)
    DEN = (1.0
           + K_CO  * p_CO
           + K_H2  * p_H2
           + K_CH4 * p_CH4
           + K_H2O * (p_H2O / p_H2))
    DEN2 = DEN ** 2

    # 5. Fuerzas impulsoras reversibles y tasas en [kmol/(kg_cat·h)]
    # R1: CH₄ + H₂O ⇌ CO + 3 H₂
    r1_kmol_h = (k1 / (p_H2 ** 2.5)) * (p_CH4 * p_H2O - (p_H2 ** 3 * p_CO) / K1) / DEN2

    # R2: CO + H₂O ⇌ CO₂ + H₂
    r2_kmol_h = (k2 / p_H2) * (p_CO * p_H2O - (p_H2 * p_CO2) / K2) / DEN2

    # R3: CH₄ + 2 H₂O ⇌ CO₂ + 4 H₂
    r3_kmol_h = (k3 / (p_H2 ** 3.5)) * (p_CH4 * (p_H2O ** 2) - (p_H2 ** 4 * p_CO2) / K3) / DEN2

    # 6. Conversión de [kmol/(kg_cat·h)] a [mol/(kg_cat·s)] y factor de eficacia
    conv_factor = (1000.0 / 3600.0) * eta_cat
    r1 = r1_kmol_h * conv_factor
    r2 = r2_kmol_h * conv_factor
    r3 = r3_kmol_h * conv_factor

    return r1, r2, r3


def reactor_odes(W, y):
    """
    Sistema de EDOs: dy/dW donde W = masa de catalizador [kg].
    y = [F_CH4, F_H2O, F_CO, F_H2, F_CO2, T]
    """
    F_CH4, F_H2O, F_CO, F_H2, F_CO2, T = y

    # Protección numérica para evitar flujos nulos o negativos
    F_CH4 = max(F_CH4, 1e-10)
    F_H2O = max(F_H2O, 1e-10)
    F_CO  = max(F_CO,  0.0)
    F_H2  = max(F_H2,  1e-8)
    F_CO2 = max(F_CO2, 0.0)

    # Flujo molar total [mol/s]
    F_total = F_CH4 + F_H2O + F_CO + F_H2 + F_CO2 + F_N2_0

    # Presiones parciales en [bar]
    p_CH4 = (F_CH4 / F_total) * P_total_bar
    p_H2O = (F_H2O / F_total) * P_total_bar
    p_CO  = (F_CO  / F_total) * P_total_bar
    p_H2  = (F_H2  / F_total) * P_total_bar
    p_CO2 = (F_CO2 / F_total) * P_total_bar

    # Velocidades de reacción [mol/(kg_cat·s)]
    r1, r2, r3 = tasas_reaccion(T, p_CH4, p_H2O, p_CO, p_H2, p_CO2)

    # CAMBIO 4: Balances molares dFi/dW incluyendo R1, R2 y R3
    dF_CH4 = -r1 - r3
    dF_H2O = -r1 - r2 - 2.0 * r3
    dF_CO  =  r1 - r2
    dF_H2  =  3.0 * r1 + r2 + 4.0 * r3
    dF_CO2 =  r2 + r3

    # Balance de energía dT/dW
    sum_FiCpi = (F_CH4 * Cp["CH4"] + F_H2O * Cp["H2O"] +
                 F_CO  * Cp["CO"]  + F_H2  * Cp["H2"]  +
                 F_CO2 * Cp["CO2"] + F_N2_0 * Cp["N2"])

    Q_ext = Ua * (T_ext - T)                             # Calor transferido del horno
    Q_rxn = -(r1 * DH_R1 + r2 * DH_R2 + r3 * DH_R3)      # Calor neto de las 3 reacciones
    dT    = (Q_ext + Q_rxn) / sum_FiCpi

    return [dF_CH4, dF_H2O, dF_CO, dF_H2, dF_CO2, dT]


# =====================================================================
#  RESOLUCIÓN DEL SISTEMA DE EDOS
# =====================================================================

y0 = [F_CH4_0, F_H2O_0, F_CO_0, F_H2_0, F_CO2_0, T0]
W_span = (0.0, W_total)
W_eval = np.linspace(0.0, W_total, 2000)

# Se utiliza Radau o BDF/LSODA dado que la cinética intrínseca de Xu & Froment es altamente rígida (stiff)
sol = solve_ivp(
    reactor_odes,
    W_span,
    y0,
    method="Radau",
    t_eval=W_eval,
    rtol=1e-6,
    atol=1e-8,
)

if not sol.success:
    print(f"AVISO: La integracion no convergio: {sol.message}")
else:
    print("OK - Integracion completada con exito.")

# Extraer resultados
W     = sol.t
F_CH4 = sol.y[0]
F_H2O = sol.y[1]
F_CO  = sol.y[2]
F_H2  = sol.y[3]
F_CO2 = sol.y[4]
T     = sol.y[5]

# Conversión de metano: X = (F_CH4,0 − F_CH4) / F_CH4,0
X_CH4 = (F_CH4_0 - F_CH4) / F_CH4_0

# Temperatura en °C
T_celsius = T - 273.15


# =====================================================================
#  GRÁFICOS
# =====================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Simulación del Reactor SMR — Modelo Cinético de (Xu & Froment, 1989)\n"
    r"R1: CH$_4$+H$_2$O $\rightleftharpoons$ CO+3H$_2$  |  "
    r"R2: CO+H$_2$O $\rightleftharpoons$ CO$_2$+H$_2$  |  "
    r"R3: CH$_4$+2H$_2$O $\rightleftharpoons$ CO$_2$+4H$_2$",
    fontsize=12, fontweight="bold",
)

# --- 1) Temperatura y Conversión vs. Masa de catalizador ---
ax1 = axes[0, 0]
color_T = "#d62728"
color_X = "#1f77b4"

ax1.set_xlabel("Masa de catalizador, W [kg]", fontsize=11)
ax1.set_ylabel("Temperatura [°C]", color=color_T, fontsize=11)
ln1 = ax1.plot(W, T_celsius, color=color_T, linewidth=2, label="T [°C]")
ax1.tick_params(axis="y", labelcolor=color_T)
ax1.set_ylim(min(T_celsius) - 20, max(T_celsius) + 20)

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
ax3.plot(W, F_CO,  label="CO",      linewidth=2)
ax3.plot(W, F_H2,  label=r"H$_2$",  linewidth=2)
ax3.plot(W, F_CO2, label=r"CO$_2$", linewidth=2)
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
plt.savefig("resultado_reactor_SMR_XuFroment.png", dpi=150, bbox_inches="tight")
plt.show()


# =====================================================================
#  RESUMEN DE RESULTADOS
# =====================================================================
print("\n" + "=" * 60)
print("  RESUMEN DE RESULTADOS DEL REACTOR SMR (Xu & Froment, 1989)")
print("=" * 60)
print(f"  Catalizador:           Ni/MgAl2O4 (eta = {eta_cat})")
print(f"  Temperatura entrada:   {T0 - 273.15:.1f} C")
print(f"  Temperatura mínima:    {min(T_celsius):.1f} C")
print(f"  Temperatura salida:    {T_celsius[-1]:.1f} C")
print(f"  Conversion CH4:        {X_CH4[-1] * 100:.2f} %")
print(f"  Presion:               {P_total_atm:.1f} atm ({P_total_bar:.2f} bar)")
print(f"  Masa de catalizador:   {W_total} kg")
print(f"  Ratio S/C (H2O/CH4):   {F_H2O_0 / F_CH4_0:.2f}")
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