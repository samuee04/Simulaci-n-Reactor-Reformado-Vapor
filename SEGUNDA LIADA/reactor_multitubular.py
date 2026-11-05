"""
REACTOR MULTITUBULAR CATALÍTICO IDEAL — Fase Gaseosa Isotérmica e Isobárica
=============================================================================
Catalizador depositado en la pared interna de cada tubo (wall-coated).

Reacción 1:  A + B  → C    (r₁' = k₁'·Cₐ·C_B)      [heterogénea]
Reacción 2:  A + C  → D    (r₂' = k₂'·Cₐ·C_C)      [heterogénea]

Variable independiente: W (masa de catalizador)
Sistema EDO:  dFⱼ/dW = rⱼ'

Estequiometría:  Cⱼ = C_T0 · (Fⱼ / F_T)

Hipótesis: T = T₀, P = P₀ constantes (reactor ideal).

Referencias: Fogler (2020), Martín (2016).
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# 1. PARÁMETROS DEL PROBLEMA  (modificar según el enunciado)
# ══════════════════════════════════════════════════════════════════════

# --- Condiciones de operación ---
T0 = 500.0          # Temperatura de entrada [K]
P0 = 10.0 * 101325  # Presión de entrada [Pa]  (10 atm → Pa)
R  = 8.314          # Constante de los gases [J/(mol·K)]  =  [Pa·m³/(mol·K)]

# --- Constantes cinéticas HETEROGÉNEAS ---
k1p = 0.5           # k₁' [m³/(mol·kg_cat·s)]  (reacción 1)
k2p = 0.2           # k₂' [m³/(mol·kg_cat·s)]  (reacción 2)

# --- Caudales molares totales de ENTRADA al reactor ---
FA0_total = 10.0    # [mol/s]  caudal molar de A
FB0_total = 10.0    # [mol/s]  caudal molar de B
FC0_total = 0.0     # [mol/s]  caudal molar de C
FD0_total = 0.0     # [mol/s]  caudal molar de D

# --- Dimensiones del reactor ---
d_t = 0.05          # Diámetro interno de cada tubo [m]
L   = 5.0           # Longitud de cada tubo [m]
N_t = 100           # Número de tubos

# --- Propiedades del catalizador ---
delta    = 0.002    # Espesor de la capa catalítica en la pared [m]
rho_cat  = 2000.0   # Densidad del catalizador [kg/m³]

# ══════════════════════════════════════════════════════════════════════
# 2. CÁLCULOS GEOMÉTRICOS Y DE CATALIZADOR
# ══════════════════════════════════════════════════════════════════════

# Volumen por tubo y total
V_tubo = (np.pi * d_t**2 * L) / 4.0      # [m³]
V_total = N_t * V_tubo                     # [m³]

# Masa de catalizador  [Fogler, 2020, pasajes 20, 62]
# W = N_t · pi · d_t · L · delta · rho_cat
W_total = N_t * np.pi * d_t * L * delta * rho_cat   # [kg]
W_tubo  = np.pi * d_t * L * delta * rho_cat          # [kg] por tubo

# Densidad aparente del lecho  [Fogler, 2020, pasajes 62, 103]
# rho_b = 4 · rho_cat · delta / d_t
rho_b = 4.0 * rho_cat * delta / d_t      # [kg/m³]

# Caudales de entrada POR TUBO
FA0 = FA0_total / N_t
FB0 = FB0_total / N_t
FC0 = FC0_total / N_t
FD0 = FD0_total / N_t

# Concentración total de entrada  [Fogler, 2020, pasajes 42, 102]
CT0 = P0 / (R * T0)    # [mol/m³]

print("=" * 65)
print("    REACTOR MULTITUBULAR CATALITICO IDEAL")
print("=" * 65)
print(f"  Temperatura de operacion  T0  = {T0:.1f} K")
print(f"  Presion de operacion      P0  = {P0/101325:.2f} atm")
print(f"  Concentracion total      CT0  = {CT0:.4f} mol/m3")
print(f"  k1' = {k1p}  m3/(mol·kg_cat·s)")
print(f"  k2' = {k2p}  m3/(mol·kg_cat·s)")
print("-" * 65)
print(f"  Diametro del tubo      d_t    = {d_t*100:.2f} cm")
print(f"  Longitud del tubo        L    = {L:.2f} m")
print(f"  Numero de tubos        N_t    = {N_t}")
print(f"  Volumen por tubo     V_tubo   = {V_tubo*1e6:.2f} cm3  ({V_tubo:.6f} m3)")
print(f"  Volumen total        V_total  = {V_total*1e3:.4f} dm3  ({V_total:.6f} m3)")
print("-" * 65)
print(f"  Espesor capa catalitica delta  = {delta*1000:.2f} mm")
print(f"  Densidad catalizador  rho_cat  = {rho_cat:.0f} kg/m3")
print(f"  Densidad aparente      rho_b   = {rho_b:.2f} kg/m3")
print(f"  Masa cat. por tubo   W_tubo   = {W_tubo:.4f} kg")
print(f"  Masa cat. total      W_total  = {W_total:.4f} kg")
print("-" * 65)
print(f"  FA0 (por tubo) = {FA0:.6f} mol/s")
print(f"  FB0 (por tubo) = {FB0:.6f} mol/s")
print(f"  FC0 (por tubo) = {FC0:.6f} mol/s")
print(f"  FD0 (por tubo) = {FD0:.6f} mol/s")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════
# 3. SISTEMA DE EDOs  (por tubo individual, variable: W)
# ══════════════════════════════════════════════════════════════════════

def sistema_EDO(W, F):
    """
    Sistema de ecuaciones diferenciales ordinarias para un tubo.
    Balance diferencial por unidad de masa catalitica.
    [Fogler, 2020, pasajes 20, 99; Martin, 2016, pasaje 178]

    Variables:
        W : masa de catalizador (variable independiente) [kg]
        F : vector [FA, FB, FC, FD] caudales molares [mol/s]

    Retorna:
        dF/dW : vector de derivadas [dFA/dW, dFB/dW, dFC/dW, dFD/dW]
    """
    FA, FB, FC, FD = F

    # Caudal molar total
    FT = FA + FB + FC + FD

    # Proteccion contra FT = 0
    if FT <= 0:
        return [0.0, 0.0, 0.0, 0.0]

    # --- Velocidades de reaccion heterogeneas ---
    # r1' = k1'·CA·CB = k1'·CT0^2·(FA/FT)·(FB/FT)
    # r2' = k2'·CA·CC = k2'·CT0^2·(FA/FT)·(FC/FT)
    # [Fogler, 2020, pasajes 48, 137]

    factor_A = CT0**2 * FA / FT

    r1p = k1p * factor_A * (FB / FT)   # velocidad reaccion 1 heterogenea
    r2p = k2p * factor_A * (FC / FT)   # velocidad reaccion 2 heterogenea

    # dFj/dW = rj'
    dFA_dW = -r1p - r2p          # rA' = -r1' - r2'
    dFB_dW = -r1p               # rB' = -r1'
    dFC_dW =  r1p - r2p         # rC' = +r1' - r2'
    dFD_dW =  r2p               # rD' = +r2'

    return [dFA_dW, dFB_dW, dFC_dW, dFD_dW]


# ══════════════════════════════════════════════════════════════════════
# 4. RESOLUCIÓN NUMÉRICA  (un tubo → luego × N_t)
# ══════════════════════════════════════════════════════════════════════

# Condiciones iniciales (por tubo, en W = 0)
# [Fogler, 2020, pasaje 135]
F0 = [FA0, FB0, FC0, FD0]

# Intervalo de integración: de W=0 a W=W_tubo
W_span = (0, W_tubo)

# Puntos de evaluación para gráficas suaves
W_eval = np.linspace(0, W_tubo, 500)

# Resolver con método RK45 (Runge-Kutta 4/5, alta precisión)
sol = solve_ivp(
    fun=sistema_EDO,
    t_span=W_span,
    y0=F0,
    method='RK45',
    t_eval=W_eval,
    rtol=1e-10,
    atol=1e-12,
    dense_output=True
)

if not sol.success:
    raise RuntimeError(f"La integracion fallo: {sol.message}")

# Extraer resultados (por tubo)
W_arr   = sol.t          # masa de catalizador [kg]
FA_arr  = sol.y[0]       # FA(W) por tubo
FB_arr  = sol.y[1]       # FB(W) por tubo
FC_arr  = sol.y[2]       # FC(W) por tubo
FD_arr  = sol.y[3]       # FD(W) por tubo

# Volumen equivalente por tubo: V = W / rho_b
V_arr = W_arr / rho_b    # [m³]

# Resultados TOTALES del reactor (todos los tubos)
FA_total_arr = FA_arr * N_t
FB_total_arr = FB_arr * N_t
FC_total_arr = FC_arr * N_t
FD_total_arr = FD_arr * N_t
W_total_arr  = W_arr * N_t    # Masa total acumulada

# Conversión de A
XA_arr = (FA0 - FA_arr) / FA0

# Selectividad instantánea de C respecto a D (donde FD > 0)
with np.errstate(divide='ignore', invalid='ignore'):
    S_CD = np.where(FD_arr > 1e-15, FC_arr / FD_arr, np.inf)

# ══════════════════════════════════════════════════════════════════════
# 5. RESULTADOS A LA SALIDA
# ══════════════════════════════════════════════════════════════════════

FA_out = FA_arr[-1]
FB_out = FB_arr[-1]
FC_out = FC_arr[-1]
FD_out = FD_arr[-1]
XA_out = XA_arr[-1]

print("\n" + "=" * 65)
print("    RESULTADOS A LA SALIDA DEL REACTOR")
print("=" * 65)
print(f"  {'Especie':<10} {'Por tubo [mol/s]':>18}  {'Total [mol/s]':>16}")
print("-" * 65)
print(f"  {'A':<10} {FA_out:>18.6f}  {FA_out*N_t:>16.6f}")
print(f"  {'B':<10} {FB_out:>18.6f}  {FB_out*N_t:>16.6f}")
print(f"  {'C':<10} {FC_out:>18.6f}  {FC_out*N_t:>16.6f}")
print(f"  {'D':<10} {FD_out:>18.6f}  {FD_out*N_t:>16.6f}")
print("-" * 65)
print(f"  Conversion de A:       X_A   = {XA_out*100:.4f} %")
if FD_out > 1e-15:
    print(f"  Selectividad C/D:    S_C/D   = {FC_out/FD_out:.4f}")
print(f"  Rendimiento de C:    Y_C     = {FC_out/FA0*100:.4f} %")
print(f"  Rendimiento de D:    Y_D     = {FD_out/FA0*100:.4f} %")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════
# 6. GRÁFICAS
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    r'Reactor Multitubular Catalítico Ideal — Resultados por tubo'
    f'\n($W_{{tubo}}$ = {W_tubo:.4f} kg,  $\\rho_b$ = {rho_b:.1f} kg/m³,  '
    f'$\\delta$ = {delta*1000:.1f} mm)',
    fontsize=13, fontweight='bold'
)

# --- Gráfica 1: Caudales molares por tubo vs W ---
ax1 = axes[0, 0]
ax1.plot(W_arr, FA_arr * 1e3, 'b-',  linewidth=2, label=r'$F_A$')
ax1.plot(W_arr, FB_arr * 1e3, 'r--', linewidth=2, label=r'$F_B$')
ax1.plot(W_arr, FC_arr * 1e3, 'g-.', linewidth=2, label=r'$F_C$')
ax1.plot(W_arr, FD_arr * 1e3, 'm:',  linewidth=2, label=r'$F_D$')
ax1.set_xlabel('Masa de catalizador por tubo, $W$ [kg]')
ax1.set_ylabel('Caudal molar [mmol/s]')
ax1.set_title('Caudales molares (por tubo)')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# --- Gráfica 2: Conversión de A vs W ---
ax2 = axes[0, 1]
ax2.plot(W_arr, XA_arr * 100, 'b-', linewidth=2)
ax2.set_xlabel('Masa de catalizador por tubo, $W$ [kg]')
ax2.set_ylabel(r'Conversión de A, $X_A$ [%]')
ax2.set_title('Conversión de A')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, min(max(XA_arr * 100) * 1.1, 105)])

# --- Gráfica 3: Concentraciones vs W ---
FT_arr = FA_arr + FB_arr + FC_arr + FD_arr
CA_arr = CT0 * FA_arr / FT_arr
CB_arr = CT0 * FB_arr / FT_arr
CC_arr = CT0 * FC_arr / FT_arr
CD_arr = CT0 * FD_arr / FT_arr

ax3 = axes[1, 0]
ax3.plot(W_arr, CA_arr, 'b-',  linewidth=2, label=r'$C_A$')
ax3.plot(W_arr, CB_arr, 'r--', linewidth=2, label=r'$C_B$')
ax3.plot(W_arr, CC_arr, 'g-.', linewidth=2, label=r'$C_C$')
ax3.plot(W_arr, CD_arr, 'm:',  linewidth=2, label=r'$C_D$')
ax3.set_xlabel('Masa de catalizador por tubo, $W$ [kg]')
ax3.set_ylabel(r'Concentración [mol/m³]')
ax3.set_title('Concentraciones')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# --- Gráfica 4: Selectividad instantánea C/D vs W ---
ax4 = axes[1, 1]
mask = np.isfinite(S_CD)
if np.any(mask):
    ax4.plot(W_arr[mask], S_CD[mask], 'g-', linewidth=2)
ax4.set_xlabel('Masa de catalizador por tubo, $W$ [kg]')
ax4.set_ylabel(r'$S_{C/D} = F_C / F_D$')
ax4.set_title('Selectividad instantánea C/D')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultados_reactor_multitubular.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGrafica guardada como 'resultados_reactor_multitubular.png'")
