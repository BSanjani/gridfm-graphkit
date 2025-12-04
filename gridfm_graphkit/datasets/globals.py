# # Global variables

# Node features indices
PD = 0
QD = 1
PG = 2
QG = 3
VM = 4
VA = 5
PQ = 6
PV = 7
REF = 8

# Edge features indices
G = 0
B = 1

# FEATURES_IDX = {"PD": PD, "QD": QD, "PG": PG, "QG": QG, "VM": VM, "VA": VA}
# BUS_TYPES = ["PQ", "PV", "REF"]

# PD_H = 0
# QD_H = 1
# QG_H = 2
# VM_H = 3
# VA_H = 4
PG_B = 5
# PQ_H = 5
# PV_H = 6
# REF_H = 7
# MIN_VM_H = 8
# MAX_VM_H = 9
# MIN_QG_H = 10
# MAX_QG_H = 11
# GS = 12
# BS = 13

# PG_H = 0
# MIN_PG = 1
# MAX_PG = 2
# C0_H = 3
# C1_H = 4
# C2_H = 5


# =========================
# === BUS FEATURE INDICES ==
# =========================
PD_H        = 0   # Active power demand (P_d)
QD_H        = 1   # Reactive power demand (Q_d)
QG_H        = 2   # Reactive power generation (Q_g)
VM_H        = 3   # Voltage magnitude (p.u.)
VA_H        = 4   # Voltage angle (degrees)
PQ_H        = 5   # PQ bus indicator (1 if PQ)
PV_H        = 6   # PV bus indicator (1 if PV)
REF_H       = 7   # Reference (slack) bus indicator (1 if REF)
MIN_VM_H    = 8   # Minimum voltage magnitude limit (p.u.)
MAX_VM_H    = 9   # Maximum voltage magnitude limit (p.u.)
MIN_QG_H    = 10  # Minimum reactive power limit (Mvar)
MAX_QG_H    = 11  # Maximum reactive power limit (Mvar)
GS          = 12  # Shunt conductance (p.u.)
BS          = 13  # Shunt susceptance (p.u.)
VN_KV       = 14  # Nominal voltage

# ================================
# === GENERATOR FEATURE INDICES ==
# ================================
PG_H        = 0   # Active power generation (P_g)
MIN_PG      = 1   # Minimum active power limit (MW)
MAX_PG      = 2   # Maximum active power limit (MW)
C0_H        = 3   # Cost coefficient c0 (€)
C1_H        = 4   # Cost coefficient c1 (€ / MW)
C2_H        = 5   # Cost coefficient c2 (€ / MW²)
G_ON        = 6   # Generator on/off

# ============================
# === EDGE FEATURE INDICES ===
# ============================
P_E         = 0   # Active power flow
Q_E         = 1   # Reactive power flow
YFF_TT_R    = 2   # Yff real
YFF_TT_I    = 3   # Yff imag
YFT_TF_R    = 4   # Yft real
YFT_TF_I    = 5   # Yft imag
TAP         = 6   # Tap ratio
ANG_MIN     = 7   # Angle min (deg)
ANG_MAX     = 8   # Angle max (deg)
RATE_A      = 9   # Thermal limit
B_ON        = 10  # Branch on/off