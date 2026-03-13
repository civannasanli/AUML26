import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson


# Gözlemlenen trafik verisi
traffic_data = np.array([12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15])

# Poisson için MLE lambda = veri ortalaması
lambda_mle = np.mean(traffic_data)

# PMF için x değerleri
x = np.arange(0, max(traffic_data) + 6)

# Her x için Poisson olasılıkları
pmf_values = poisson.pmf(x, lambda_mle)

# Grafik boyutu
plt.figure(figsize=(10, 6))

# Gerçek veri histogramı
plt.hist(
    traffic_data,
    bins=np.arange(min(traffic_data) - 0.5, max(traffic_data) + 1.5, 1),
    density=True,
    alpha=0.6,
    label="Gerçek Veri Histogramı"
)

# Poisson PMF grafiği
plt.plot(
    x,
    pmf_values,
    'o-',
    label=f"Poisson PMF (λ = {lambda_mle:.3f})"
)

# Grafik detayları
plt.xlabel("1 Dakikadaki Araç Sayısı")
plt.ylabel("Olasılık")
plt.title("Trafik Verisi Histogramı ve Poisson PMF Karşılaştırması")
plt.legend()
plt.grid(True, alpha=0.3)

# Grafiği göster
plt.show()