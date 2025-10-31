# Global variables

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

FEATURES_IDX = {"PD": PD, "QD": QD, "PG": PG, "QG": QG, "VM": VM, "VA": VA}
BUS_TYPES = ["PQ", "PV", "REF"]

PD_H = 0
QD_H = 1
QG_H = 2
VM_H = 3
VA_H = 4
PG_B = 5
PQ_H = 5
PV_H = 6
REF_H = 7
MIN_VM_H = 8
MAX_VM_H = 9
MIN_QG_H = 10
MAX_QG_H = 11

PG_H = 0
MIN_PG = 1
MAX_PG = 2
C0_H = 3
C1_H = 4
C2_H = 5