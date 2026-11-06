"""
REACTOR MULTITUBULAR CATALÍTICO — Fase Gaseosa NO ISOTÉRMICA con Calor Externo
================================================================================
Catalizador depositado en la pared interna de cada tubo (wall-coated).

Estequiometría del sistema reactivo:
  Reacción 1:  A + B ⇌ C + 3D    (r₁' = k₁'·[C_A·C_B − C_C·C_D³ / K_eq,1])
  Reacción 2:  C + B ⇌ E + D     (r₂' = k₂'·[C_C·C_B − C_E·C_D   / K_eq,2])

  (Típico del reformado con vapor: A=CH₄, B=H₂O, C=CO, D=H₂, E=CO₂)

Variable independiente: W (masa de catalizador)
Sistema de EDOs acoplado:
    dF_A/dW = −r₁'
    dF_B/dW = −r₁' − r₂'
    dF_C/dW =  r₁' − r₂'
    dF_D/dW = 3r₁' + r₂'
    dF_E/dW =  r₂'

    dT/dW   = [Q̇_transf + Σ(−ΔH_rxn,r · r_r')] / [Σ Fⱼ · Cp,j(T)]
              donde Q̇_transf = U · a · (T_a − T)

Transferencia de calor externa:  Q̇ = U · a · (Tₐ − T)
    donde a = área de transferencia por unidad de masa de catalizador [m²/kg]

Estequiometría:  Cⱼ = (Fⱼ / F_T) · (P / R·T)

Hipótesis:
    - P = P₀ constante (isobárico)
    - La cinética sigue Arrhenius: k(T) = k_ref · exp[E/R · (1/T_ref − 1/T)]
    - Las constantes de equilibrio siguen van 't Hoff:
        K_eq(T) = K_eq(T_ref) · exp[−ΔH_rxn/R · (1/T − 1/T_ref)]

Referencias: Fogler (2020), Xu & Froment (1989), Martín (2016).

══════════════════════════════════════════════════════════════════════
  LEYENDA DE VARIABLES
══════════════════════════════════════════════════════════════════════
  Q̇_ext    : Potencia térmica cedida al tubo por los quemadores/horno [W o kW]
  T_in     : Temperatura de entrada al reactor (~500–600 °C → 773–873 K)
  T_out    : Temperatura de salida del reactor (~800–850 °C → 1073–1123 K)
  U        : Coeficiente global de transferencia de calor [W/(m²·K)]
  T_a      : Temperatura de los gases de combustión externos [K]  (constante)
  ξ_r      : Avance molar total de la reacción r:  ξ_r = ΔF_{j,r} / ν_{j,r}
  Σ Fⱼ·Cpⱼ(T) : Capacidad calorífica total de la mezcla a temperatura local T
══════════════════════════════════════════════════════════════════════
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# 1. PARÁMETROS DEL PROBLEMA  (modificar según el enunciado)
# ══════════════════════════════════════════════════════════════════════

# --- Condiciones de operación ---
T0   = 823.15        # Temperatura de entrada [K]  (~550 °C)
P0   = 10.0 * 101325 # Presión de entrada [Pa]  (10 atm → Pa)
R_g  = 8.314         # Constante de los gases [J/(mol·K)]

# --- Constantes cinéticas HETEROGÉNEAS (a temperatura de referencia) ---
T_ref = 823.15       # Temperatura de referencia para k y K_eq [K] (= T0)
k1p_ref = 0.5        # k₁'(T_ref) [m³/(mol·kg_cat·s)]   (reacción 1)
k2p_ref = 0.2        # k₂'(T_ref) [m³/(mol·kg_cat·s)]   (reacción 2)

# Energías de activación [J/mol]
# ╔══════════════════════════════════════════════════════════════════╗
# ║  VALORES PLACEHOLDER — sustituir por datos reales del sistema  ║
# ╚══════════════════════════════════════════════════════════════════╝
E1 = 80000.0         # Energía de activación reacción 1 [J/mol]
E2 = 60000.0         # Energía de activación reacción 2 [J/mol]

# Constantes de equilibrio químico a T_ref
# Para rxn 1: K_eq,1 = (C_C · C_D³) / (C_A · C_B)  →  unidades [(mol/m³)²]
# Para rxn 2: K_eq,2 = (C_E · C_D)  / (C_C · C_B)  →  adimensional [-]
K_eq1_ref = 5.0e4    # K_eq,1(T_ref) [(mol/m³)²]
K_eq2_ref = 1.5      # K_eq,2(T_ref) [-]

# --- Entalpías de reacción (a T_ref) ---
# ╔══════════════════════════════════════════════════════════════════╗
# ║  VALORES PLACEHOLDER — sustituir por datos reales del sistema  ║
# ║  Positivo = endotérmica (consume calor)                        ║
# ║  Negativo = exotérmica (libera calor)                          ║
# ╚══════════════════════════════════════════════════════════════════╝
DH_rxn1_ref =  206000.0  # ΔH_rxn1(T_ref) [J/mol]  (endotérmica, reformado)
DH_rxn2_ref =  -41000.0  # ΔH_rxn2(T_ref) [J/mol]  (exotérmica moderada, shift)

# --- Caudales molares totales de ENTRADA al reactor ---
FA0_total = 10.0     # [mol/s]  caudal molar de A (ej. CH4)
FB0_total = 30.0     # [mol/s]  caudal molar de B (ej. H2O, relación vapor/carbono = 3:1)
FC0_total = 0.0      # [mol/s]  caudal molar de C (ej. CO)
FD0_total = 0.0      # [mol/s]  caudal molar de D (ej. H2)
FE0_total = 0.0      # [mol/s]  caudal molar de E (ej. CO2)

# --- Dimensiones del reactor ---
d_t = 0.05           # Diámetro interno de cada tubo [m]
L   = 5.0            # Longitud de cada tubo [m]
N_t = 100            # Número de tubos

# --- Propiedades del catalizador ---
delta   = 0.002      # Espesor de la capa catalítica en la pared [m]
rho_cat = 2000.0     # Densidad del catalizador [kg/m³]

# --- Transferencia de calor externa ---
U   = 50.0           # Coeficiente global de transf. de calor [W/(m²·K)]
T_a = 1173.15        # Temperatura de los gases de combustión [K] (~900 °C)

# ══════════════════════════════════════════════════════════════════════
# 2. CAPACIDADES CALORÍFICAS  Cp(T) = a + b·T + c·T²   [J/(mol·K)]
# ══════════════════════════════════════════════════════════════════════
#
# Coeficientes polinómicos genéricos (tipo NIST/Shomate simplificado)
# ╔══════════════════════════════════════════════════════════════════╗
# ║  VALORES PLACEHOLDER — sustituir por datos reales de cada      ║
# ║  especie concreta del sistema de reacción.                     ║
# ║  Formato:  Cp_j(T) = a_j + b_j·T + c_j·T²  [J/(mol·K)]      ║
# ╚══════════════════════════════════════════════════════════════════╝
#
#                   a [J/(mol·K)]    b [J/(mol·K²)]     c [J/(mol·K³)]
Cp_coefs = {
    'A': np.array([28.0,            1.5e-2,            -0.5e-5]),  # Especie A (ej. CH4)
    'B': np.array([32.0,            1.0e-2,            -0.2e-5]),  # Especie B (ej. H2O)
    'C': np.array([29.0,            0.8e-2,            -0.1e-5]),  # Especie C (ej. CO)
    'D': np.array([27.0,            0.5e-2,            -0.1e-5]),  # Especie D (ej. H2)
    'E': np.array([40.0,            2.5e-2,            -1.0e-5]),  # Especie E (ej. CO2)
}


def Cp_j(species, T):
    """Capacidad calorífica molar de la especie j a temperatura T [J/(mol·K)]."""
    a, b, c = Cp_coefs[species]
    return a + b * T + c * T**2


def DH_rxn_T(DH_ref, T, T_ref_val, Delta_nu_Cp_coefs):
    """
    Entalpía de reacción corregida con la temperatura (ley de Kirchhoff):
        ΔH_rxn(T) = ΔH_rxn(T_ref) + ∫[T_ref→T] Σ(νⱼ·Cpⱼ) dT

    Delta_nu_Cp_coefs = [Δa, Δb, Δc] donde Δa = Σ νⱼ·aⱼ, etc.
    """
    Da, Db, Dc = Delta_nu_Cp_coefs
    dT = T - T_ref_val
    return DH_ref + Da * dT + (Db / 2.0) * (T**2 - T_ref_val**2) + (Dc / 3.0) * (T**3 - T_ref_val**3)


# Coeficientes Σ(νⱼ·Cpⱼ) para cada reacción (ley de Kirchhoff)
# Reacción 1:  A + B ⇌ C + 3D    →  ν_A=-1, ν_B=-1, ν_C=+1, ν_D=+3
# Reacción 2:  C + B ⇌ E + D     →  ν_C=-1, ν_B=-1, ν_E=+1, ν_D=+1
Delta_Cp_coefs_rxn1 = (  1 * Cp_coefs['C']
                       + 3 * Cp_coefs['D']
                       - 1 * Cp_coefs['A']
                       - 1 * Cp_coefs['B'])

Delta_Cp_coefs_rxn2 = (  1 * Cp_coefs['E']
                       + 1 * Cp_coefs['D']
                       - 1 * Cp_coefs['C']
                       - 1 * Cp_coefs['B'])


# ══════════════════════════════════════════════════════════════════════
# 3. CÁLCULOS GEOMÉTRICOS Y DE CATALIZADOR
# ══════════════════════════════════════════════════════════════════════

# Volumen por tubo y total
V_tubo  = (np.pi * d_t**2 * L) / 4.0      # [m³]
V_total = N_t * V_tubo                      # [m³]

# Masa de catalizador
W_total = N_t * np.pi * d_t * L * delta * rho_cat   # [kg]
W_tubo  = np.pi * d_t * L * delta * rho_cat          # [kg] por tubo

# Densidad aparente del lecho
rho_b = 4.0 * rho_cat * delta / d_t   # [kg/m³]

# Área lateral interna por tubo (superficie de transferencia de calor)
A_lat_tubo = np.pi * d_t * L   # [m²]

# Área de transferencia por unidad de masa de catalizador [m²/kg_cat]
a_especifica = A_lat_tubo / W_tubo  # [m²/kg_cat]

# Caudales de entrada POR TUBO
FA0 = FA0_total / N_t
FB0 = FB0_total / N_t
FC0 = FC0_total / N_t
FD0 = FD0_total / N_t
FE0 = FE0_total / N_t

# Concentración total de entrada  C_T0 = P0 / (R·T0)
CT0 = P0 / (R_g * T0)    # [mol/m³]

print("=" * 70)
print("    REACTOR MULTITUBULAR CATALÍTICO — NO ISOTÉRMICO")
print("    Reacciones: A + B <-> C + 3D ; C + B <-> E + D")
print("    Balance de Energía con Calentamiento Externo")
print("=" * 70)
print(f"  Temperatura de entrada     T_in  = {T0:.2f} K  ({T0-273.15:.1f} °C)")
print(f"  Presión de operación       P0    = {P0/101325:.2f} atm")
print(f"  Concentración total        CT0   = {CT0:.4f} mol/m³")
print(f"  k1'(T_ref) = {k1p_ref}  m³/(mol·kg_cat·s)   E1 = {E1/1000:.1f} kJ/mol")
print(f"  k2'(T_ref) = {k2p_ref}  m³/(mol·kg_cat·s)   E2 = {E2/1000:.1f} kJ/mol")
print(f"  K_eq,1(T_ref) = {K_eq1_ref:.2e} (mol/m³)²")
print(f"  K_eq,2(T_ref) = {K_eq2_ref:.2f}")
print("-" * 70)
print(f"  DH_rxn1(T_ref) = {DH_rxn1_ref/1000:.1f} kJ/mol  (endotermica)")
print(f"  DH_rxn2(T_ref) = {DH_rxn2_ref/1000:.1f} kJ/mol  (exotermica)")
print("-" * 70)
print(f"  Diametro del tubo        d_t     = {d_t*100:.2f} cm")
print(f"  Longitud del tubo          L     = {L:.2f} m")
print(f"  Número de tubos          N_t     = {N_t}")
print(f"  Volumen por tubo       V_tubo    = {V_tubo*1e6:.2f} cm³  ({V_tubo:.6f} m³)")
print(f"  Volumen total          V_total   = {V_total*1e3:.4f} dm³  ({V_total:.6f} m³)")
print("-" * 70)
print(f"  Espesor capa catalitica  delta   = {delta*1000:.2f} mm")
print(f"  Densidad catalizador   rho_cat   = {rho_cat:.0f} kg/m³")
print(f"  Densidad aparente        rho_b   = {rho_b:.2f} kg/m³")
print(f"  Masa cat. por tubo     W_tubo    = {W_tubo:.4f} kg")
print(f"  Masa cat. total        W_total   = {W_total:.4f} kg")
print("-" * 70)
print(f"  Area lateral por tubo  A_lat     = {A_lat_tubo:.4f} m2")
print(f"  Area especifica          a       = {a_especifica:.4f} m2/kg_cat")
print("-" * 70)
print(f"  U (coef. global)                 = {U:.1f} W/(m2*K)")
print(f"  T_a (gases combustion)           = {T_a:.2f} K  ({T_a-273.15:.1f} C)")
print("-" * 70)
print(f"  FA0 (por tubo) = {FA0:.6f} mol/s")
print(f"  FB0 (por tubo) = {FB0:.6f} mol/s")
print(f"  FC0 (por tubo) = {FC0:.6f} mol/s")
print(f"  FD0 (por tubo) = {FD0:.6f} mol/s")
print(f"  FE0 (por tubo) = {FE0:.6f} mol/s")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# 4. SISTEMA DE EDOs ACOPLADO  (por tubo individual, variable: W)
# ══════════════════════════════════════════════════════════════════════

def sistema_EDO(W, y):
    """
    Sistema de ecuaciones diferenciales acoplado:
    Balances de materia + Balance de energía para un tubo.

    Variables:
        W : masa de catalizador (variable independiente) [kg]
        y : vector [FA, FB, FC, FD, FE, T]
            FA, FB, FC, FD, FE : caudales molares [mol/s]
            T                  : temperatura local [K]

    Retorna:
        dy/dW : [dFA/dW, dFB/dW, dFC/dW, dFD/dW, dFE/dW, dT/dW]
    """
    FA, FB, FC, FD, FE, T = y

    # Evitar concentraciones o temperaturas no físicas
    FA = max(FA, 0.0)
    FB = max(FB, 0.0)
    FC = max(FC, 0.0)
    FD = max(FD, 0.0)
    FE = max(FE, 0.0)
    T  = max(T, 100.0)

    # Caudal molar total
    FT = FA + FB + FC + FD + FE

    # Protección contra FT = 0
    if FT <= 1e-15:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # ─── Concentración total local (gas ideal, isobárico) ───
    CT = P0 / (R_g * T)    # [mol/m³]

    # Concentraciones molares individuales [mol/m³]
    CA = CT * (FA / FT)
    CB = CT * (FB / FT)
    CC = CT * (FC / FT)
    CD = CT * (FD / FT)
    CE = CT * (FE / FT)

    # ─── Constantes cinéticas con Arrhenius ───
    # k(T) = k(T_ref) · exp[E/R · (1/T_ref − 1/T)]
    k1p = k1p_ref * np.exp((E1 / R_g) * (1.0 / T_ref - 1.0 / T))
    k2p = k2p_ref * np.exp((E2 / R_g) * (1.0 / T_ref - 1.0 / T))

    # ─── Constantes de equilibrio químico (van 't Hoff) ───
    # K_eq(T) = K_eq(T_ref) · exp[-ΔH/R · (1/T − 1/T_ref)]
    K_eq1 = K_eq1_ref * np.exp((-DH_rxn1_ref / R_g) * (1.0 / T - 1.0 / T_ref))
    K_eq2 = K_eq2_ref * np.exp((-DH_rxn2_ref / R_g) * (1.0 / T - 1.0 / T_ref))

    # Proteger contra división por cero en K_eq
    K_eq1 = max(K_eq1, 1e-10)
    K_eq2 = max(K_eq2, 1e-10)

    # ─── Velocidades de reacción heterogéneas reversibles ───
    # Reacción 1: A + B ⇌ C + 3D
    # r₁' = k₁' · [CA·CB − (CC · CD³) / K_eq,1]
    r1p = k1p * (CA * CB - (CC * (CD**3)) / K_eq1)

    # Reacción 2: C + B ⇌ E + D
    # r₂' = k₂' · [CC·CB − (CE · CD) / K_eq,2]
    r2p = k2p * (CC * CB - (CE * CD) / K_eq2)

    # ─── Balances de materia: dFⱼ/dW = rⱼ' ───
    # A + B ⇌ C + 3D
    # C + B ⇌ E + D
    dFA_dW = -r1p
    dFB_dW = -r1p - r2p
    dFC_dW =  r1p - r2p
    dFD_dW = 3.0 * r1p + r2p
    dFE_dW =  r2p

    # ─── Balance de energía: dT/dW ───
    # Calor transferido desde el exterior [W/kg_cat = J/(s·kg_cat)]
    Q_dot = U * a_especifica * (T_a - T)

    # Entalpías de reacción corregidas con T (ley de Kirchhoff)
    DH_rxn1 = DH_rxn_T(DH_rxn1_ref, T, T_ref, Delta_Cp_coefs_rxn1)
    DH_rxn2 = DH_rxn_T(DH_rxn2_ref, T, T_ref, Delta_Cp_coefs_rxn2)

    # Término de generación/consumo de calor por reacción [W/kg_cat]
    Q_rxn = (-DH_rxn1) * r1p + (-DH_rxn2) * r2p

    # Capacidad calorífica total de la mezcla [W/K = J/(s·K)]
    Sum_FjCpj = (FA * Cp_j('A', T)
               + FB * Cp_j('B', T)
               + FC * Cp_j('C', T)
               + FD * Cp_j('D', T)
               + FE * Cp_j('E', T))

    # Derivada de temperatura
    if abs(Sum_FjCpj) < 1e-30:
        dT_dW = 0.0
    else:
        dT_dW = (Q_dot + Q_rxn) / Sum_FjCpj

    return [dFA_dW, dFB_dW, dFC_dW, dFD_dW, dFE_dW, dT_dW]


# ══════════════════════════════════════════════════════════════════════
# 5. RESOLUCIÓN NUMÉRICA  (un tubo → luego × N_t)
# ══════════════════════════════════════════════════════════════════════

# Condiciones iniciales (por tubo, en W = 0)
y0 = [FA0, FB0, FC0, FD0, FE0, T0]

# Intervalo de integración: de W=0 a W=W_tubo
W_span = (0, W_tubo)

# Puntos de evaluación para gráficas suaves
W_eval = np.linspace(0, W_tubo, 500)

# Resolver con método Radau (implícito, robusto para sistemas acoplados stiff)
print("\nResolviendo sistema de EDOs acoplado (materia + energia)...")
sol = solve_ivp(
    fun=sistema_EDO,
    t_span=W_span,
    y0=y0,
    method='Radau',
    t_eval=W_eval,
    rtol=1e-8,
    atol=1e-10,
    dense_output=True
)
print(f"  Integracion completada. Exito: {sol.success}")

if not sol.success:
    raise RuntimeError(f"La integración falló: {sol.message}")

# Extraer resultados (por tubo)
W_arr   = sol.t          # masa de catalizador [kg]
FA_arr  = sol.y[0]       # FA(W) por tubo
FB_arr  = sol.y[1]       # FB(W) por tubo
FC_arr  = sol.y[2]       # FC(W) por tubo
FD_arr  = sol.y[3]       # FD(W) por tubo
FE_arr  = sol.y[4]       # FE(W) por tubo
T_arr   = sol.y[5]       # T(W)  temperatura local [K]

# Volumen equivalente por tubo: V = W / rho_b
V_arr = W_arr / rho_b    # [m³]

# Resultados TOTALES del reactor (todos los tubos)
FA_total_arr = FA_arr * N_t
FB_total_arr = FB_arr * N_t
FC_total_arr = FC_arr * N_t
FD_total_arr = FD_arr * N_t
FE_total_arr = FE_arr * N_t
W_total_arr  = W_arr * N_t

# Conversiones de los reactivos
XA_arr = (FA0 - FA_arr) / FA0
XB_arr = (FB0 - FB_arr) / FB0

# ══════════════════════════════════════════════════════════════════════
# 6. CÁLCULOS ENERGÉTICOS Y AVANCES MOLARES GLOBALES
# ══════════════════════════════════════════════════════════════════════

# Avances molares totales de cada reacción (por tubo):
#   FA = FA0 − ξ₁           →  ξ₁ = FA0 − FA
#   FE = FE0 + ξ₂           →  ξ₂ = FE − FE0
xi_1_tubo = FA0 - FA_arr[-1]
xi_2_tubo = FE_arr[-1] - FE0

xi_1_total = xi_1_tubo * N_t    # [mol/s]  avance total rxn 1
xi_2_total = xi_2_tubo * N_t    # [mol/s]  avance total rxn 2

# Potencia térmica transferida al tubo (integrada numéricamente)
Q_perfil = U * a_especifica * (T_a - T_arr)   # [W/kg_cat] perfil
Q_ext_tubo = np.trapezoid(Q_perfil, W_arr)         # [W] por tubo
Q_ext_total = Q_ext_tubo * N_t                  # [W] total reactor

# Potencia consumida por las reacciones
DH1_out = DH_rxn_T(DH_rxn1_ref, T_arr[-1], T_ref, Delta_Cp_coefs_rxn1)
DH2_out = DH_rxn_T(DH_rxn2_ref, T_arr[-1], T_ref, Delta_Cp_coefs_rxn2)
Q_rxn_total = (DH1_out * xi_1_total + DH2_out * xi_2_total)  # [W]

# ══════════════════════════════════════════════════════════════════════
# 7. RESULTADOS A LA SALIDA
# ══════════════════════════════════════════════════════════════════════

FA_out = FA_arr[-1]
FB_out = FB_arr[-1]
FC_out = FC_arr[-1]
FD_out = FD_arr[-1]
FE_out = FE_arr[-1]
XA_out = XA_arr[-1]
XB_out = XB_arr[-1]
T_out  = T_arr[-1]

print("\n" + "=" * 70)
print("    RESULTADOS A LA SALIDA DEL REACTOR")
print("=" * 70)
print(f"  {'Especie':<10} {'Por tubo [mol/s]':>18}  {'Total [mol/s]':>16}")
print("-" * 70)
print(f"  {'A':<10} {FA_out:>18.6f}  {FA_out*N_t:>16.6f}")
print(f"  {'B':<10} {FB_out:>18.6f}  {FB_out*N_t:>16.6f}")
print(f"  {'C':<10} {FC_out:>18.6f}  {FC_out*N_t:>16.6f}")
print(f"  {'D':<10} {FD_out:>18.6f}  {FD_out*N_t:>16.6f}")
print(f"  {'E':<10} {FE_out:>18.6f}  {FE_out*N_t:>16.6f}")
print("-" * 70)
print(f"  Conversión de A:        X_A   = {XA_out*100:.4f} %")
print(f"  Conversión de B:        X_B   = {XB_out*100:.4f} %")
print(f"  Rendimiento de C (vs A): Y_C  = {FC_out/FA0*100:.4f} %")
print(f"  Rendimiento de D (vs A): Y_D  = {FD_out/(3.0*FA0)*100:.4f} %")
print(f"  Rendimiento de E (vs A): Y_E  = {FE_out/FA0*100:.4f} %")
print("-" * 70)
print(f"  Temperatura entrada:   T_in   = {T0:.2f} K  ({T0-273.15:.1f} °C)")
print(f"  Temperatura salida:    T_out  = {T_out:.2f} K  ({T_out-273.15:.1f} °C)")
print(f"  Incremento de T:        ΔT    = {T_out-T0:.2f} K")
print("-" * 70)
print(f"  Avance molar rxn 1:     ξ₁    = {xi_1_total:.6f} mol/s")
print(f"  Avance molar rxn 2:     ξ₂    = {xi_2_total:.6f} mol/s")
print("-" * 70)
print(f"  Q̇_ext total (quemadores):     = {Q_ext_total/1000:.4f} kW")
print(f"  Q̇_rxn total (reacciones):     = {Q_rxn_total/1000:.4f} kW")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# 8. GRÁFICAS  (4 paneles exactos solicitados)
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Reactor Multitubular Catalítico — No Isotérmico con Calor Externo\n'
    r'Reacciones: $A + B \rightleftharpoons C + 3D$  ;  $C + B \rightleftharpoons E + D$' + '\n'
    rf'$W_{{tubo}}$ = {W_tubo:.4f} kg,  $\rho_b$ = {rho_b:.1f} kg/m³,  '
    rf'$U$ = {U:.0f} W/(m²·K),  $T_a$ = {T_a-273.15:.0f} °C',
    fontsize=13, fontweight='bold'
)

# --- Gráfica 1: Conversión de los reactivos vs W ---
ax1 = axes[0, 0]
ax1.plot(W_arr, XA_arr * 100, 'b-',  linewidth=2, label=r'$X_A$')
ax1.plot(W_arr, XB_arr * 100, 'r--', linewidth=2, label=r'$X_B$')
ax1.set_xlabel('Masa de catalizador por tubo, $W$ [kg]')
ax1.set_ylabel('Conversión [%]')
ax1.set_title('Conversión de los reactivos')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
max_X = max(float(np.max(XA_arr)), float(np.max(XB_arr)))
ax1.set_ylim([0, min(max(max_X * 110, 10), 105)])

# --- Gráfica 2: Fracción molar de cada compuesto vs W ---
FT_arr = FA_arr + FB_arr + FC_arr + FD_arr + FE_arr
yA_arr = FA_arr / FT_arr
yB_arr = FB_arr / FT_arr
yC_arr = FC_arr / FT_arr
yD_arr = FD_arr / FT_arr
yE_arr = FE_arr / FT_arr

ax2 = axes[0, 1]
ax2.plot(W_arr, yA_arr, 'b-',  linewidth=2, label=r'$y_A$')
ax2.plot(W_arr, yB_arr, 'r--', linewidth=2, label=r'$y_B$')
ax2.plot(W_arr, yC_arr, 'g-.', linewidth=2, label=r'$y_C$')
ax2.plot(W_arr, yD_arr, 'm:',  linewidth=2, label=r'$y_D$')
ax2.plot(W_arr, yE_arr, 'c-',  linewidth=2, label=r'$y_E$')
ax2.set_xlabel('Masa de catalizador por tubo, $W$ [kg]')
ax2.set_ylabel('Fracción molar, $y_j$')
ax2.set_title('Fracción molar de cada compuesto a lo largo del reactor')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.05])

# --- Gráfica 3: Perfil de temperatura vs W ---
ax3 = axes[1, 0]
ax3.plot(W_arr, T_arr - 273.15, 'r-', linewidth=2.5, label=r'$T(W)$')
ax3.axhline(y=T_a - 273.15, color='orange', linestyle='--', linewidth=1.5,
            label=rf'$T_a$ = {T_a-273.15:.0f} °C (gases combustión)')
ax3.axhline(y=T0 - 273.15, color='blue', linestyle=':', linewidth=1.5,
            label=rf'$T_{{in}}$ = {T0-273.15:.0f} °C')
ax3.set_xlabel('Masa de catalizador por tubo, $W$ [kg]')
ax3.set_ylabel('Temperatura [°C]')
ax3.set_title('Variación de temperatura del reactor a lo largo de este')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# --- Gráfica 4: Flujo de calor externo vs W ---
Q_perfil_kW = Q_perfil / 1000.0   # [kW/kg_cat]
ax4 = axes[1, 1]
ax4.plot(W_arr, Q_perfil_kW, 'darkorange', linewidth=2.5,
         label=r'$\dot{Q}_{ext} = U \cdot a \cdot (T_a - T)$')
ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax4.set_xlabel('Masa de catalizador por tubo, $W$ [kg]')
ax4.set_ylabel(r'$\dot{Q}_{ext}$ [kW/kg$_{cat}$]')
ax4.set_title('Flujo de calor externo')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultados_reactor_no_isotermico.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGráfica guardada como 'resultados_reactor_no_isotermico.png'")
