from scipy.io import loadmat


def main():
    dmos_data = loadmat("dmos.mat")
    print("=== Inhalt von dmos.mat ===")
    print(dmos_data.keys())

    dmos = dmos_data["dmos"].flatten()
    orgs = dmos_data["orgs"].flatten()

    print("Anzahl Einträge:", len(dmos))
    print("Erste 10 DMOS-Werte:", dmos[:10])
    print("Erste 10 ORGS-Werte:", orgs[:10])

    refnames_data = loadmat("refnames_all.mat")
    print("\n=== Inhalt von refnames_all.mat ===")
    print(refnames_data.keys())

    refnames_all = refnames_data["refnames_all"]
    print("Shape von refnames_all:", refnames_all.shape)

    names_array = refnames_all[0]

    print("Anzahl Namen:", len(names_array))
    print("Erste 10 Referenznamen:")

    max_count = min(10, len(names_array))
    for i in range(max_count):
        name = names_array[i][0]
        print(f"{i}: {name}")

    example_index = 2
    print("\n--- Beispielindex ---")
    print("Index:", example_index)
    print("DMOS:", dmos[example_index])
    print("ORGS (1=Ref, 0=Distorted):", orgs[example_index])
    print("Referenzbildname:", names_array[example_index][0])


if __name__ == "__main__":
    main()
