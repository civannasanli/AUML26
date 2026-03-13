import numpy as np


# Orijinal trafik verisi
traffic_data = np.array([12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15])

# Outlier eklenmiş veri
traffic_data_outlier = np.append(traffic_data, 200)

# Eski ve yeni lambda değerleri
old_lambda = np.mean(traffic_data)
new_lambda = np.mean(traffic_data_outlier)

print("=== Bölüm 4: Outlier Analizi ===")
print(f"Outlier öncesi lambda: {old_lambda:.6f}")
print(f"Outlier sonrası lambda: {new_lambda:.6f}")
print(f"Fark: {new_lambda - old_lambda:.6f}")