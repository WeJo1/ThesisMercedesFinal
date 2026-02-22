from scipy.io import loadmat

# ==========================
# 1. dmos.mat anschauen
# ==========================

dmos_data = loadmat("dmos.mat")
print("=== Inhalt von dmos.mat ===")
print(dmos_data.keys())

# Arrays aus der .mat-Datei holen
dmos = dmos_data["dmos"].flatten()   # DMOS-Werte
orgs = dmos_data["orgs"].flatten()   # 1 = Referenz, 0 = verzerrt

print("Anzahl Einträge:", len(dmos))
print("Erste 10 DMOS-Werte:", dmos[:10])
print("Erste 10 ORGS-Werte:", orgs[:10])

# ==========================
# 2. refnames_all.mat anschauen
# ==========================

refnames_data = loadmat("refnames_all.mat")
print("\n=== Inhalt von refnames_all.mat ===")
print(refnames_data.keys())

refnames_all = refnames_data["refnames_all"]
print("Shape von refnames_all:", refnames_all.shape)

# In LIVE ist refnames_all ein Array der Form (1, N),
# also 1 Zeile mit N Spalten. Wir holen uns diese eine Zeile.
names_array = refnames_all[0]

print("Anzahl Namen:", len(names_array))
print("Erste 10 Referenznamen:")
for i in range(min(10, len(names_array))):
    # Jeder Eintrag ist ein verschachteltes Array, das den String enthält.
    # Meist ist das Pattern: names_array[i][0]
    name = names_array[i][0]
    print(f"{i}: {name}")

# ==========================
# 3. Beispiel: Index i anschauen
# ==========================

# Wenn du z.B. einen bestimmten Index überprüfen willst:
example_index = 2
print("\n--- Beispielindex ---")
print("Index:", example_index)
print("DMOS:", dmos[example_index])
print("ORGS (1=Ref, 0=Distorted):", orgs[example_index])
print("Referenzbildname:", names_array[example_index][0])
