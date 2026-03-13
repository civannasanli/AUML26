import numpy as np
import scipy.optimize as opt


# Gözlemlenen trafik verisi
traffic_data = np.array([12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15])


def negative_log_likelihood(lam, data):
    """
    Poisson dağılımı için negatif log-likelihood hesaplar.

    Parametreler:
    lam  -> optimizer tarafından gelen lambda değeri
    data -> gözlem verisi

    Dönüş:
    Negatif log-likelihood değeri
    """

    # scipy.optimize.minimize bazen parametreyi tek sayılık dizi olarak yollar
    if isinstance(lam, (list, np.ndarray)):
        lam = lam[0]

    # Poisson parametresi lambda pozitif olmak zorundadır
    if lam <= 0:
        return np.inf

    # Veri sayısını bulur
    n = len(data)

    # Poisson için log(k!) terimi lambda'ya bağlı olmadığı için sabit kabul edilip atılabilir
    # Negatif log-likelihood:
    # NLL = n * lam - sum(data) * log(lam)
    nll = n * lam - np.sum(data) * np.log(lam)

    return nll


def numerical_mle(data):
    """
    Sayısal optimizasyon ile MLE lambda tahminini bulur.
    """
    # Başlangıç tahmini
    initial_guess = [1.0]

    # Optimizasyon işlemi
    result = opt.minimize(
        negative_log_likelihood,
        initial_guess,
        args=(data,),
        bounds=[(0.001, None)]
    )

    return result.x[0]


def analytical_mle(data):
    """
    Poisson dağılımı için analitik MLE sonucu veri ortalamasıdır.
    """
    return np.mean(data)


if __name__ == "__main__":
    lambda_numerical = numerical_mle(traffic_data)
    lambda_analytical = analytical_mle(traffic_data)

    print("=== Bölüm 2: Python ile Sayısal MLE ===")
    print(f"Sayısal Tahmin (MLE lambda): {lambda_numerical:.6f}")
    print(f"Analitik Tahmin (Ortalama): {lambda_analytical:.6f}")