"""
==========================================================================
 SIMULACIÓN INDUSTRIAL DE UN REACTOR PFR MULTITUBULAR DE REFORMADO (SMR)
 Modelo Cinético LHHW de (Xu & Froment, 1989) con Difusión Interna
==========================================================================
 Configuración industrial:
   - Catalizador: Ni/MgAl2O4 en pastillas/anillos comerciales (eta = 0.05)
   - Haz multitubular: N_tubos = 32 tubos en paralelo (D_in = 0.10 m, L ≈ 12 m)
   - Temperatura límite metalúrgica en pared: T_ext = 850 °C

 Reacciones (Xu & Froment, 1989):
   R1: CH₄ + H₂O   ⇌  CO  + 3 H₂     (Reformado a CO — endotérmica)
   R2: CO  + H₂O   ⇌  CO₂ + H₂       (Water-Gas Shift — exotérmica)
   R3: CH₄ + 2 H₂O ⇌  CO₂ + 4 H₂     (Reformado a CO₂ — endotérmica)
==========================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =====================================================================
#  PARÁMETROS MODIFICABLES
# =====================================================================
# --- Flujos molares de entrada [kmol/h] ---
F_CH4_0_kmol_h = 60.0      # Metano
F_H2O_0_kmol_h = 200.0     # Vapor de agua (H2O/CH4 = 3.33; H2O/C_tot = 2.0)
F_CO_0_kmol_h  = 0.0       # Monóxido de carbono
F_H2_0_kmol_h  = 1.0       # Hidrógeno (traza necesaria para evitar singularidad)
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
P_total_bar = P_total_atm * 1.01325  # Presión total [bar] (20.265 bar)

# --- CAMBIO 1: Factor de eficacia industrial y geometría multitubular ---
W_total = 3000.0                     # Masa total de catalizador [kg]
eta_cat = 0.05                       # Factor de eficacia por difusión interna (5%)

N_tubos = 32                         # Número de tubos en paralelo en el horno
D_tubo  = 0.10                       # Diámetro interno de cada tubo [m]
rho_bed = 1000.0                     # Densidad aparente del lecho [kg/m³]

# Sección transversal de un tubo y del haz completo [m²]
A_tubo  = np.pi * (D_tubo ** 2) / 4.0
A_total = N_tubos * A_tubo           # Área de paso total = 0.2513 m²
dW_dz   = rho_bed * A_total          # Carga lineal de catalizador = 251.33 kg/m
L_total = W_total / dW_dz            # Longitud de cada tubo = 11.94 m (≈ 12 m)

# --- CAMBIO 2: Límite metalúrgico de temperatura externa de pared ---
U_coef   = 300.0                     # Coeficiente global U [J/(s·m²·K)]
area_esp = 4.0 / (D_tubo * rho_bed)  # Área específica [m²/kg_cat] = 0.04
Ua       = U_coef * area_esp         # Coeficiente volumétrico [J/(s·kg_cat·K)] = 12.0
T_ext    = 850.0 + 273.15            # Temperatura máxima operativa de pared [K] (850 °C)

# --- Constante universal de gases ---
R_GAS = 8.314                        # [J/(mol·K)]

# =====================================================================
#  PARÁMETROS CINÉTICOS Y DE ADSORCIÓN (Xu & Froment, 1989)
# =====================================================================
# --- Constantes cinéticas intrínsecas: k_i = A_i * exp(-Ea_i / (R * T)) ---
A1  = 4.225e15     # [kmol · bar^0.5 / (kg_cat · h)]
Ea1 = 240.1e3      # [J/mol]

A2  = 1.955e6      # [kmol / (kg_cat · h · bar)]
Ea2 = 67.13e3      # [J/mol]

A3  = 1.020e15     # [kmol · bar^0.5 / (kg_cat · h)]
Ea3 = 243.9e3      # [J/mol]

# --- Constantes de adsorción: K_j = A_j * exp(-dH_ads_j / (R * T)) ---
A_CO   = 8.23e-5   # [bar^-1]
dH_CO  = -70.65e3  # [J/mol]

A_H2   = 6.12e-9   # [bar^-1]
dH_H2  = -82.90e3  # [J/mol]

A_CH4  = 6.65e-4   # [bar^-1]
dH_CH4 = -38.28e3  # [J/mol]

A_H2O  = 1.77e5    # [adimensional]
dH_H2O = 88.68e3   # [J/mol]

# --- Constantes de equilibrio termodinámico ---
Kp1_a = 30.420
Kp1_b = 27_106.0   # [K]
Kp2_a = -3.798
Kp2_b = 4_160.0    # [K]

# --- Entalpías de reacción [J/mol] ---
DH_R1 =  206.1e3   # ΔH₁ (endotérmica)
DH_R2 =  -41.15e3  # ΔH₂ (exotérmica)
DH_R3 =  164.9e3   # ΔH₃ (endotérmica)

# --- Capacidades caloríficas promedio [J/(mol·K)] ---
Cp = {
    "CH4": 76.5,
    "H2O": 41.5,
    "CO":  30.5,
    "H2":  29.0,
    "CO2": 45.0,
    "N2":  30.0,
}


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
    Calcula las velocidades efectivas r₁, r₂ y r₃ en [mol/(kg_cat·s)]
    aplicando el factor de eficacia catalítica eta_cat = 0.05.
    """
    k1 = A1 * np.exp(-Ea1 / (R_GAS * T))
    k2 = A2 * np.exp(-Ea2 / (R_GAS * T))
    k3 = A3 * np.exp(-Ea3 / (R_GAS * T))

    K_CO  = A_CO  * np.exp(-dH_CO  / (R_GAS * T))
    K_H2  = A_H2  * np.exp(-dH_H2  / (R_GAS * T))
    K_CH4 = A_CH4 * np.exp(-dH_CH4 / (R_GAS * T))
    K_H2O = A_H2O * np.exp(-dH_H2O / (R_GAS * T))

    K1 = K1_eq(T)
    K2 = K2_eq(T)
    K3 = K3_eq(T)

    DEN = (1.0
           + K_CO  * p_CO
           + K_H2  * p_H2
           + K_CH4 * p_CH4
           + K_H2O * (p_H2O / p_H2))
    DEN2 = DEN ** 2

    # Tasas intrínsecas en [kmol/(kg_cat·h)]
    r1_kmol_h = (k1 / (p_H2 ** 2.5)) * (p_CH4 * p_H2O - (p_H2 ** 3 * p_CO) / K1) / DEN2
    r2_kmol_h = (k2 / p_H2) * (p_CO * p_H2O - (p_H2 * p_CO2) / K2) / DEN2
    r3_kmol_h = (k3 / (p_H2 ** 3.5)) * (p_CH4 * (p_H2O ** 2) - (p_H2 ** 4 * p_CO2) / K3) / DEN2

    # Conversión a [mol/(kg_cat·s)] multiplicando por eta_cat
    conv_factor = (1000.0 / 3600.0) * eta_cat
    return r1_kmol_h * conv_factor, r2_kmol_h * conv_factor, r3_kmol_h * conv_factor


def reactor_odes(W, y):
    """
    Sistema de EDOs: dy/dW donde W = masa de catalizador [kg].
    y = [F_CH4, F_H2O, F_CO, F_H2, F_CO2, T]
    """
    F_CH4, F_H2O, F_CO, F_H2, F_CO2, T = y

    F_CH4 = max(F_CH4, 1e-10)
    F_H2O = max(F_H2O, 1e-10)
    F_CO  = max(F_CO,  0.0)
    F_H2  = max(F_H2,  1e-8)
    F_CO2 = max(F_CO2, 0.0)

    F_total = F_CH4 + F_H2O + F_CO + F_H2 + F_CO2 + F_N2_0

    p_CH4 = (F_CH4 / F_total) * P_total_bar
    p_H2O = (F_H2O / F_total) * P_total_bar
    p_CO  = (F_CO  / F_total) * P_total_bar
    p_H2  = (F_H2  / F_total) * P_total_bar
    p_CO2 = (F_CO2 / F_total) * P_total_bar

    r1, r2, r3 = tasas_reaccion(T, p_CH4, p_H2O, p_CO, p_H2, p_CO2)

    dF_CH4 = -r1 - r3
    dF_H2O = -r1 - r2 - 2.0 * r3
    dF_CO  =  r1 - r2
    dF_H2  =  3.0 * r1 + r2 + 4.0 * r3
    dF_CO2 =  r2 + r3

    sum_FiCpi = (F_CH4 * Cp["CH4"] + F_H2O * Cp["H2O"] +
                 F_CO  * Cp["CO"]  + F_H2  * Cp["H2"]  +
                 F_CO2 * Cp["CO2"] + F_N2_0 * Cp["N2"])

    Q_ext = Ua * (T_ext - T)
    Q_rxn = -(r1 * DH_R1 + r2 * DH_R2 + r3 * DH_R3)
    dT    = (Q_ext + Q_rxn) / sum_FiCpi

    return [dF_CH4, dF_H2O, dF_CO, dF_H2, dF_CO2, dT]


# =====================================================================
#  RESOLUCIÓN DEL SISTEMA DE EDOS
# =====================================================================

y0 = [F_CH4_0, F_H2O_0, F_CO_0, F_H2_0, F_CO2_0, T0]
W_span = (0.0, W_total)
W_eval = np.linspace(0.0, W_total, 2000)

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
z_len = W / dW_dz            # CAMBIO 3: Vector de longitud axial del tubo [m]
F_CH4 = sol.y[0]
F_H2O = sol.y[1]
F_CO  = sol.y[2]
F_H2  = sol.y[3]
F_CO2 = sol.y[4]
T     = sol.y[5]

X_CH4     = (F_CH4_0 - F_CH4) / F_CH4_0
T_celsius = T - 273.15


# =====================================================================
#  GRÁFICOS (Representados en función de la longitud axial z [m])
# =====================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"Reactor SMR Multitubular ({N_tubos} tubos, L = {L_total:.2f} m) — "
    f"Xu & Froment (1989) con η = {eta_cat} y T_pared = {T_ext - 273.15:.0f} °C",
    fontsize=12, fontweight="bold",
)

# --- 1) Temperatura y Conversión vs. Longitud del reactor z [m] ---
ax1 = axes[0, 0]
color_T = "#d62728"
color_X = "#1f77b4"

ax1.set_xlabel("Longitud axial del tubo, z [m]", fontsize=11)
ax1.set_ylabel("Temperatura [°C]", color=color_T, fontsize=11)
ln1 = ax1.plot(z_len, T_celsius, color=color_T, linewidth=2, label="T [°C]")
ax1.tick_params(axis="y", labelcolor=color_T)
ax1.set_ylim(min(T_celsius) - 20, max(T_celsius) + 20)

ax1b = ax1.twinx()
ax1b.set_ylabel("Conversión CH₄ (X)", color=color_X, fontsize=11)
ln2 = ax1b.plot(z_len, X_CH4, color=color_X, linewidth=2, linestyle="--", label="X CH₄")
ax1b.tick_params(axis="y", labelcolor=color_X)
ax1b.set_ylim(-0.02, 1.02)

# Eje superior secundario con la masa de catalizador W [kg]
ax1_top = ax1.twiny()
ax1_top.set_xlim(ax1.get_xlim()[0] * dW_dz, ax1.get_xlim()[1] * dW_dz)
ax1_top.set_xlabel("Masa acumulada de catalizador, W [kg]", fontsize=9, color="gray")

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right", fontsize=10)
ax1.set_title("Perfiles Axiales de Temperatura y Conversión", fontsize=11)
ax1.grid(True, alpha=0.3)

# --- 2) Conversión vs. Temperatura (gráfico paramétrico) ---
ax2 = axes[0, 1]
ax2.plot(T_celsius, X_CH4, color="#2ca02c", linewidth=2.5)
ax2.set_xlabel("Temperatura [°C]", fontsize=11)
ax2.set_ylabel("Conversión CH₄ (X)", fontsize=11)
ax2.set_title("Trayectoria Conversión vs. Temperatura", fontsize=11)
ax2.set_ylim(-0.02, 1.02)
ax2.grid(True, alpha=0.3)
ax2.annotate(
    "Entrada (z = 0 m)",
    xy=(T_celsius[0], X_CH4[0]),
    fontsize=9, fontweight="bold",
    xytext=(-40, 20), textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="gray"),
)
ax2.annotate(
    f"Salida (z = {L_total:.1f} m)",
    xy=(T_celsius[-1], X_CH4[-1]),
    fontsize=9, fontweight="bold",
    xytext=(-70, -25), textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="gray"),
)

# --- 3) Flujos molares totales vs. Longitud axial z [m] ---
ax3 = axes[1, 0]
ax3.plot(z_len, F_CH4, label=r"CH$_4$", linewidth=2)
ax3.plot(z_len, F_H2O, label=r"H$_2$O", linewidth=2)
ax3.plot(z_len, F_CO,  label="CO",      linewidth=2)
ax3.plot(z_len, F_H2,  label=r"H$_2$",  linewidth=2)
ax3.plot(z_len, F_CO2, label=r"CO$_2$", linewidth=2)
ax3.set_xlabel("Longitud axial del tubo, z [m]", fontsize=11)
ax3.set_ylabel("Flujo molar total (32 tubos) [mol/s]", fontsize=11)
ax3.set_title("Evolución de Flujos Molares vs. Longitud z", fontsize=11)
ax3.legend(fontsize=10, ncol=2)
ax3.grid(True, alpha=0.3)

# --- 4) Composición molar a la salida (base húmeda) ---
ax4 = axes[1, 1]
especies = [r"CH$_4$", r"H$_2$O", "CO", r"H$_2$", r"CO$_2$", r"N$_2$"]
flujos_salida = [F_CH4[-1], F_H2O[-1], F_CO[-1], F_H2[-1], F_CO2[-1], F_N2_0]
F_total_sal = sum(flujos_salida)
fracciones = [f / F_total_sal * 100 for f in flujos_salida]
colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

bars = ax4.bar(especies, fracciones, color=colores, edgecolor="black", linewidth=0.5)
ax4.set_ylabel("Fracción molar húmeda [%]", fontsize=11)
ax4.set_title("Composición del Gas de Síntesis a la Salida", fontsize=11)
ax4.grid(True, axis="y", alpha=0.3)
for bar, frac in zip(bars, fracciones):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{frac:.1f}%",
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("resultado_reactor_SMR_industrial.png", dpi=150, bbox_inches="tight")
plt.show()


# =====================================================================
#  RESUMEN DE RESULTADOS INDUSTRIALES
# =====================================================================
idx_Tmin = np.argmin(T_celsius)
print("\n" + "=" * 65)
print("  RESUMEN DEL REACTOR SMR MULTITUBULAR INDUSTRIAL (Xu & Froment)")
print("=" * 65)
print(f"  Configuración tubular: {N_tubos} tubos (D_in = {D_tubo:.2f} m, L = {L_total:.2f} m)")
print(f"  Catalizador total:     {W_total:.0f} kg Ni/MgAl2O4 (eta_cat = {eta_cat})")
print(f"  Tiempo espacial ef.:   {(W_total * 1000 / (F_CH4_0_kmol_h * 1000)) * eta_cat:.2f} g_cat·h/mol_CH4")
print(f"  Temperatura entrada:   {T0 - 273.15:.1f} °C")
print(f"  Pozo térmico (T_min):  {T_celsius[idx_Tmin]:.1f} °C (en z = {z_len[idx_Tmin]:.2f} m)")
print(f"  Temperatura pared:     {T_ext - 273.15:.1f} °C")
print(f"  Temperatura salida:    {T_celsius[-1]:.1f} °C")
print(f"  Conversión de CH4:     {X_CH4[-1] * 100:.2f} %")
print(f"  Presión de operación:  {P_total_atm:.1f} atm ({P_total_bar:.2f} bar)")
print("-" * 65)
print("  Flujos molares a la salida [mol/s] (y fracción molar húmeda [%]):")
print(f"    CH4  = {F_CH4[-1]:8.4f} mol/s  ({F_CH4[-1]/F_total_sal*100:5.2f} %)")
print(f"    H2O  = {F_H2O[-1]:8.4f} mol/s  ({F_H2O[-1]/F_total_sal*100:5.2f} %)")
print(f"    CO   = {F_CO[-1]:8.4f} mol/s  ({F_CO[-1]/F_total_sal*100:5.2f} %)")
print(f"    H2   = {F_H2[-1]:8.4f} mol/s  ({F_H2[-1]/F_total_sal*100:5.2f} %)")
print(f"    CO2  = {F_CO2[-1]:8.4f} mol/s  ({F_CO2[-1]/F_total_sal*100:5.2f} %)")
print(f"    TOTAL= {F_total_sal:8.4f} mol/s  (100.00 %)")
print("=" * 65)