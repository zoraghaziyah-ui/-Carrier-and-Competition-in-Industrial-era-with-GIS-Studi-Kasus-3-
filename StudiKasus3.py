import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print("=== 1. MEMUAT DATASET SPASIAL ASET PROPERTI ===")
np.random.seed(42)
n_assets = 200

data_aset = {
    'Asset_ID': [f"AST-{i:03d}" for i in range(1, n_assets + 1)],
    'Latitude': np.random.uniform(-3.8200, -3.7800, size=n_assets),
    'Longitude': np.random.uniform(102.2500, 102.3000, size=n_assets),
    'Asset_Value_USD': np.random.randint(50000, 500000, size=n_assets),
    'Maintenance_Status': np.random.choice(['Good', 'Needs Repair', 'Critical'], size=n_assets, p=[0.6, 0.3, 0.1])
}

df_gis = pd.DataFrame(data_aset)
print("5 Baris Pertama Data Aset GIS:")
print(df_gis.head(), "\n")

print("=== 2. ANALISIS KLUSTER WILAYAH PEMELIHARAAN ASET ===")
X = df_gis[['Latitude', 'Longitude']]

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_gis['Zone_Cluster'] = kmeans.fit_predict(X)

print("Jumlah aset di setiap zona kluster pemeliharaan:")
print(df_gis['Zone_Cluster'].value_counts(), "\n")

print("=== 3. RINGKASAN NILAI ASET PER ZONA ===")
summary_zone = df_gis.groupby('Zone_Cluster').agg(
    Total_Aset=('Asset_ID', 'count'),
    Total_Nilai_Aset=('Asset_Value_USD', 'sum'),
    Rata_Rata_Nilai=('Asset_Value_USD', 'mean')
).reset_index()

print(summary_zone)

df_gis.to_csv("gis_asset_clustered_results.csv", index=False)
print("\nHasil analisis spasial berhasil disimpan ke 'gis_asset_clustered_results.csv'")