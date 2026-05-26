import numpy as np

# Data mahasiswa dari file Excel (6 fitur numerik)
data = [
    ("Dzaky Prima Yoga", [4, 4, 3, 4, 4, 5]),
    ("Salwa Okta Herliana", [2, 2, 2, 4, 4, 5]),
    ("Gizalsa Syabilla", [1, 1, 1, 5, 4, 5]),
    ("Revaldy Dearly Zaliandy", [3, 3, 3, 5, 4, 4]),
    ("Anas", [3, 2, 2, 4, 3, 3]),
    ("Belva Risma Aydina", [5, 5, 4, 5, 5, 5]),
    ("Zahid Abdul Majid", [2, 1, 1, 5, 3, 3]),
    ("Zahra'Ateltline Mufidah", [2, 2, 2, 4, 5, 4]),
    ("Zakiyah Tsaniyah Siregar", [3, 2, 2, 4, 5, 5]),
    ("RATUU", [3, 2, 1, 4, 5, 3]),
    ("Azka Ayudya W", [3, 2, 3, 5, 4, 5]),
    ("Nayla Tiani Putri", [3, 4, 3, 4, 3, 3]),
    ("Djaka", [4, 4, 3, 1, 3, 3]),
    ("Ridwan Nur Zidan", [2, 2, 3, 4, 2, 3]),
    ("Nazhif", [5, 3, 4, 5, 3, 5]),
    ("Zaskia", [3, 3, 2, 4, 3, 3]),
    ("Khairatul Husna Tartila", [3, 3, 2, 4, 3, 4]),
    ("Ihdaaa", [3, 2, 2, 5, 3, 3]),
    ("Danish Febrianto Putra", [2, 2, 2, 1, 4, 5]),
    ("Elsy Sabmita Dilijani", [2, 2, 1, 4, 5, 5]),
    ("Dhita Paramita Maryadin", [4, 4, 3, 5, 4, 4]),
    ("Muhammad Romdhon Indra Galuh", [1, 5, 2, 4, 1, 3]),
    ("Aisyah Nur Rahmah", [4, 3, 2, 4, 5, 5]),
    ("Vicky Adriano", [4, 3, 4, 2, 4, 4]),
    ("Adhwa Tsabitah", [3, 2, 1, 5, 4, 4]),
    ("Imelda", [3, 2, 2, 5, 3, 4]),
    ("Faradeswita Azzahra", [2, 5, 2, 5, 3, 3])
]

X = np.array([d[1] for d in data])
jumlah_cluster = 3

# Centroid awal: 3 mahasiswa pertama
centroid = np.array([
    data[0][1],  # Dzaky Prima Yoga
    data[1][1],  # Salwa Okta Herliana
    data[2][1]   # Gizalsa Syabilla
])

def tentukan_cluster(X, centroid):
    cluster = []
    for x in X:
        jarak = np.linalg.norm(x - centroid, axis=1)
        cluster.append(np.argmin(jarak))
    return np.array(cluster)

def perbarui_centroid(X, cluster, jumlah_cluster, centroid_lama):
    centroid_baru = []
    for i in range(jumlah_cluster):
        anggota = X[cluster == i]
        if len(anggota) > 0:
            centroid_baru.append(anggota.mean(axis=0))
        else:
            centroid_baru.append(centroid_lama[i])
    return np.array(centroid_baru)

maks_iterasi = 10
for i in range(maks_iterasi):
    print(f"\n--------------- Iterasi {i + 1} ---------------")
    cluster = tentukan_cluster(X, centroid)

    # tampilkan anggota cluster
    for id_cluster in range(jumlah_cluster):
        anggota = [data[idx][0] for idx in np.where(cluster == id_cluster)[0]]
        print(f"Cluster {id_cluster + 1} (Total {len(anggota)}): {', '.join(anggota)}")

    # tampilkan centroid sebelum update
    print("\n-- Centroid Sebelum Update --")
    for j in range(jumlah_cluster):
        nilai = ', '.join(f'{v:.2f}' for v in centroid[j])
        print(f"Centroid {j + 1}: [{nilai}]")

    centroid_baru = perbarui_centroid(X, cluster, jumlah_cluster, centroid)

    # tampilkan centroid setelah update
    print("\n-- Centroid Setelah Update --")
    for j in range(jumlah_cluster):
        nilai = ', '.join(f'{v:.2f}' for v in centroid_baru[j])
        print(f"Centroid {j + 1}: [{nilai}]")

    if np.allclose(centroid, centroid_baru):
        print("\nCentroid sudah konvergen, iterasi berhenti.")
        break

    centroid = centroid_baru

print("\n======= Hasil Clustering Akhir =======\n")
for i in range(jumlah_cluster):
    print(f"Cluster {i + 1} (Jumlah Anggota: {np.sum(cluster == i)}):")
    for idx in np.where(cluster == i)[0]:
        print(f"  - {data[idx][0]} | {data[idx][1]}")
    nilai_centroid = ', '.join(f'{v:.2f}' for v in centroid[i])
    print(f"Centroid Cluster {i + 1}: [{nilai_centroid}]\n")