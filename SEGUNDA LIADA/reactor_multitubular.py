"""
REACTOR MULTITUBULAR IDEAL — Fase Gaseosa Isotérmica e Isobárica
=================================================================
Reacción 1:  A + B  → C    (r₁ = k₁·Cₐ·C_B)
Reacción 2:  A + C  → D    (r₂ = k₂·Cₐ·C_C)

Sistema EDO:  dFⱼ/dV = rⱼ   (por tubo)
Estequiometría:  Cⱼ = C_T0 · (Fⱼ / F_T)

Hipótesis: T = T₀, P = P₀ constantes (reactor ideal).
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

# --- Constantes cinéticas ---
k1 = 0.5            # [m³/(mol·s)]   (reacción 1)
k2 = 0.2            # [m³/(mol·s)]   (reacción 2)

# --- Caudales molares totales de ENTRADA al reactor ---
FA0_total = 10.0    # [mol/s]  caudal molar de A
FB0_total = 10.0    # [mol/s]  caudal molar de B
FC0_total = 0.0     # [mol/s]  caudal molar de C
FD0_total = 0.0     # [mol/s]  caudal molar de D

# --- Dimensiones del reactor ---
d_t = 0.05          # Diámetro interno de cada tubo [m]
L   = 5.0           # Longitud de cada tubo [m]
N_t = 100           # Número de tubos

# ══════════════════════════════════════════════════════════════════════
# 2. CÁLCULOS GEOMÉTRICOS
# ══════════════════════════════════════════════════════════════════════

V_tubo = (np.pi * d_t**2 * L) / 4.0      # Volumen por tubo  [m³]
V_total = N_t * V_tubo                     # Volumen total     [m³]

# Caudales de entrada POR TUBO
FA0 = FA0_total / N_t
FB0 = FB0_total / N_t
FC0 = FC0_total / N_t
FD0 = FD0_total / N_t

# Concentración total de entrada
CT0 = P0 / (R * T0)    # [mol/m³]

print("=" * 60)
print("        REACTOR MULTITUBULAR IDEAL")
print("=" * 60)
print(f"  Temperatura de operación  T₀ = {T0:.1f} K")
print(f"  Presión de operación      P₀ = {P0/101325:.2f} atm")
print(f"  Concentración total      CT₀ = {CT0:.4f} mol/m³")
print(f"  k₁ = {k1}  m³/(mol·s)")
print(f"  k₂ = {k2}  m³/(mol·s)")
print("-" * 60)
print(f"  Diámetro del tubo    d_t = {d_t*100:.2f} cm")
print(f"  Longitud del tubo      L = {L:.2f} m")
print(f"  Número de tubos      N_t = {N_t}")
print(f"  Volumen por tubo   V_tubo = {V_tubo*1e6:.2f} cm³  ({V_tubo:.6f} m³)")
print(f"  Volumen total      V_total = {V_total*1e3:.4f} dm³  ({V_total:.6f} m³)")
print("-" * 60)
print(f"  FA₀ (por tubo) = {FA0:.6f} mol/s")
print(f"  FB₀ (por tubo) = {FB0:.6f} mol/s")
print(f"  FC₀ (por tubo) = {FC0:.6f} mol/s")
print(f"  FD₀ (por tubo) = {FD0:.6f} mol/s")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# 3. SISTEMA DE EDOs  (por tubo individual)
# ══════════════════════════════════════════════════════════════════════

def sistema_EDO(V, F):
    """
    Sistema de ecuaciones diferenciales ordinarias para un tubo.
    
    Variables:
        V : volumen del reactor (variable independiente) [m³]
        F : vector [FA, FB, FC, FD] caudales molares [mol/s]
    
    Retorna:
        dF/dV : vector de derivadas [dFA/dV, dFB/dV, dFC/dV, dFD/dV]
    """
    FA, FB, FC, FD = F

    # Caudal molar total
    FT = FA + FB + FC + FD

    # Protección contra FT = 0
    if FT <= 0:
        return [0.0, 0.0, 0.0, 0.0]

    # --- Velocidades de reacción netas ---
    # r₁ = k₁·Cₐ·C_B = k₁·CT₀²·(FA/FT)·(FB/FT)
    # r₂ = k₂·Cₐ·C_C = k₂·CT₀²·(FA/FT)·(FC/FT)

    factor_A = CT0**2 * FA / FT

    r1 = k1 * factor_A * (FB / FT)   # velocidad reacción 1
    r2 = k2 * factor_A * (FC / FT)   # velocidad reacción 2

    # dFⱼ/dV = rⱼ
    dFA_dV = -r1 - r2           # rA = -r₁ - r₂
    dFB_dV = -r1                # rB = -r₁
    dFC_dV =  r1 - r2           # rC = +r₁ - r₂
    dFD_dV =  r2                # rD = +r₂

    return [dFA_dV, dFB_dV, dFC_dV, dFD_dV]


# ══════════════════════════════════════════════════════════════════════
# 4. RESOLUCIÓN NUMÉRICA  (un tubo → luego × N_t)
# ══════════════════════════════════════════════════════════════════════

# Condiciones iniciales (por tubo)
F0 = [FA0, FB0, FC0, FD0]

# Intervalo de integración: de V=0 a V=V_tubo
V_span = (0, V_tubo)

# Puntos de evaluación para gráficas suaves
V_eval = np.linspace(0, V_tubo, 500)

# Resolver con método RK45 (Runge-Kutta 4/5, alta precisión)
sol = solve_ivp(
    fun=sistema_EDO,
    t_span=V_span,
    y0=F0,
    method='RK45',
    t_eval=V_eval,
    rtol=1e-10,
    atol=1e-12,
    dense_output=True
)

if not sol.success:
    raise RuntimeError(f"La integración falló: {sol.message}")

# Extraer resultados (por tubo)
V_arr   = sol.t          # volumen [m³]
FA_arr  = sol.y[0]       # FA(V) por tubo
FB_arr  = sol.y[1]       # FB(V) por tubo
FC_arr  = sol.y[2]       # FC(V) por tubo
FD_arr  = sol.y[3]       # FD(V) por tubo

# Resultados TOTALES del reactor (todos los tubos)
FA_total = FA_arr * N_t
FB_total = FB_arr * N_t
FC_total = FC_arr * N_t
FD_total = FD_arr * N_t
V_total_arr = V_arr * N_t   # Volumen acumulado total

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

print("\n" + "=" * 60)
print("        RESULTADOS A LA SALIDA DEL REACTOR")
print("=" * 60)
print(f"  {'Especie':<10} {'Por tubo [mol/s]':>18}  {'Total [mol/s]':>16}")
print("-" * 60)
print(f"  {'A':<10} {FA_out:>18.6f}  {FA_out*N_t:>16.6f}")
print(f"  {'B':<10} {FB_out:>18.6f}  {FB_out*N_t:>16.6f}")
print(f"  {'C':<10} {FC_out:>18.6f}  {FC_out*N_t:>16.6f}")
print(f"  {'D':<10} {FD_out:>18.6f}  {FD_out*N_t:>16.6f}")
print("-" * 60)
print(f"  Conversión de A:       X_A = {XA_out*100:.4f} %")
if FD_out > 1e-15:
    print(f"  Selectividad C/D:    S_C/D = {FC_out/FD_out:.4f}")
print(f"  Rendimiento de C:    Y_C   = {FC_out/FA0*100:.4f} %")
print(f"  Rendimiento de D:    Y_D   = {FD_out/FA0*100:.4f} %")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# 6. GRÁFICAS
# ══════════════════════════════════════════════════════════════════════

# Convertir volumen a cm³ para mejor legibilidad
V_plot = V_arr * 1e6  # m³ → cm³

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Reactor Multitubular Ideal — Resultados por tubo', fontsize=14, fontweight='bold')

# --- Gráfica 1: Caudales molares por tubo ---
ax1 = axes[0, 0]
ax1.plot(V_plot, FA_arr * 1e3, 'b-',  linewidth=2, label=r'$F_A$')
ax1.plot(V_plot, FB_arr * 1e3, 'r--', linewidth=2, label=r'$F_B$')
ax1.plot(V_plot, FC_arr * 1e3, 'g-.', linewidth=2, label=r'$F_C$')
ax1.plot(V_plot, FD_arr * 1e3, 'm:',  linewidth=2, label=r'$F_D$')
ax1.set_xlabel('Volumen por tubo [cm³]')
ax1.set_ylabel('Caudal molar [mmol/s]')
ax1.set_title('Caudales molares (por tubo)')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# --- Gráfica 2: Conversión de A ---
ax2 = axes[0, 1]
ax2.plot(V_plot, XA_arr * 100, 'b-', linewidth=2)
ax2.set_xlabel('Volumen por tubo [cm³]')
ax2.set_ylabel('Conversión de A [%]')
ax2.set_title('Conversión de A')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, max(XA_arr * 100) * 1.1])

# --- Gráfica 3: Concentraciones ---
FT_arr = FA_arr + FB_arr + FC_arr + FD_arr
CA_arr = CT0 * FA_arr / FT_arr
CB_arr = CT0 * FB_arr / FT_arr
CC_arr = CT0 * FC_arr / FT_arr
CD_arr = CT0 * FD_arr / FT_arr

ax3 = axes[1, 0]
ax3.plot(V_plot, CA_arr, 'b-',  linewidth=2, label=r'$C_A$')
ax3.plot(V_plot, CB_arr, 'r--', linewidth=2, label=r'$C_B$')
ax3.plot(V_plot, CC_arr, 'g-.', linewidth=2, label=r'$C_C$')
ax3.plot(V_plot, CD_arr, 'm:',  linewidth=2, label=r'$C_D$')
ax3.set_xlabel('Volumen por tubo [cm³]')
ax3.set_ylabel('Concentración [mol/m³]')
ax3.set_title('Concentraciones')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# --- Gráfica 4: Selectividad instantánea C/D ---
ax4 = axes[1, 1]
# Filtrar valores infinitos para la gráfica
mask = np.isfinite(S_CD)
if np.any(mask):
    ax4.plot(V_plot[mask], S_CD[mask], 'g-', linewidth=2)
ax4.set_xlabel('Volumen por tubo [cm³]')
ax4.set_ylabel(r'$S_{C/D}$ = $F_C / F_D$')
ax4.set_title('Selectividad instantánea C/D')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultados_reactor_multitubular.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Gráfica guardada como 'resultados_reactor_multitubular.png'")
