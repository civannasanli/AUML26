import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
true_mu = 150.0     # Gerçek parlaklık
true_sigma = 10.0   # Gözlem hatası
n_obs = 50          # Gözlem sayısı
np.random.seed(42)
data = true_mu + true_sigma * np.random.randn(n_obs)

print("=" * 55)
print("   YZM212 - 4. Ödev: Galaksi Parlaklık Analizi")
print("=" * 55)
print(f"\n[VERİ] {n_obs} adet sentetik gözlem oluşturuldu.")
print(f"  Veri Ortalaması  : {data.mean():.4f}")
print(f"  Veri Std. Sapma  : {data.std():.4f}")


def log_likelihood(theta, data):
    mu, sigma = theta
    if sigma <= 0:
        return -np.inf
    return -0.5 * np.sum(((data - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))


def log_prior(theta):
    mu, sigma = theta
    if 0 < mu < 300 and 0 < sigma < 50:
        return 0.0
    return -np.inf


def log_probability(theta, data):
    """Log-Posterior = Log-Prior + Log-Likelihood"""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data)


initial = [140, 5]
n_walkers = 32
n_steps = 2000
burn_in = 500

pos = initial + 1e-4 * np.random.randn(n_walkers, 2)

print(f"\n[MCMC] {n_walkers} walker, {n_steps} adım çalıştırılıyor...")
sampler = emcee.EnsembleSampler(n_walkers, 2, log_probability, args=(data,))
sampler.run_mcmc(pos, n_steps, progress=True)

# Burn-in atılarak örnekler alınıyor
flat_samples = sampler.get_chain(discard=burn_in, thin=15, flat=True)
print(f"\n[MCMC] Tamamlandı. Toplam örnek: {flat_samples.shape[0]}")
mu_samples = flat_samples[:, 0]
sigma_samples = flat_samples[:, 1]

# Posterior istatistikleri
mu_median = np.median(mu_samples)
sigma_median = np.median(sigma_samples)

mu_low, mu_high = np.percentile(mu_samples, [16, 84])
sigma_low, sigma_high = np.percentile(sigma_samples, [16, 84])

mu_err = (mu_high - mu_low) / 2
sigma_err = (sigma_high - sigma_low) / 2

mu_abs_error = abs(mu_median - true_mu)
sigma_abs_error = abs(sigma_median - true_sigma)

print("\n" + "=" * 55)
print("   POSTERIOR SONUÇLARI")
print("=" * 55)
print(f"\n  μ (Parlaklık):")
print(f"    Gerçek Değer   : {true_mu:.1f}")
print(f"    Tahmin (Median): {mu_median:.4f}")
print(f"    %16 Sınırı    : {mu_low:.4f}")
print(f"    %84 Sınırı    : {mu_high:.4f}")
print(f"    Belirsizlik   : ±{mu_err:.4f}")
print(f"    Mutlak Hata   : {mu_abs_error:.4f}")

print(f"\n  σ (Hata Payı):")
print(f"    Gerçek Değer   : {true_sigma:.1f}")
print(f"    Tahmin (Median): {sigma_median:.4f}")
print(f"    %16 Sınırı    : {sigma_low:.4f}")
print(f"    %84 Sınırı    : {sigma_high:.4f}")
print(f"    Belirsizlik   : ±{sigma_err:.4f}")
print(f"    Mutlak Hata   : {sigma_abs_error:.4f}")

print("\n" + "=" * 55)
print("   PARAMETRE KARŞILAŞTIRMA TABLOSU (Bölüm 5.1)")
print("=" * 55)
print(f"{'Değişken':<15} {'Gerçek':>8} {'Median':>10} {'Alt(%16)':>10} {'Üst(%84)':>10} {'Mut.Hata':>10}")
print("-" * 65)
print(
    f"{'μ (Parlaklık)':<15} {true_mu:>8.1f} {mu_median:>10.4f} {mu_low:>10.4f} {mu_high:>10.4f} {mu_abs_error:>10.4f}")
print(
    f"{'σ (Hata Payı)':<15} {true_sigma:>8.1f} {sigma_median:>10.4f} {sigma_low:>10.4f} {sigma_high:>10.4f} {sigma_abs_error:>10.4f}")


fig1, ax = plt.subplots(figsize=(8, 4))
ax.hist(data, bins=12, color='steelblue', edgecolor='white', alpha=0.85, label='Gözlem Verisi')
ax.axvline(true_mu, color='red', lw=2, linestyle='--', label=f'Gerçek μ = {true_mu}')
ax.axvline(mu_median, color='orange', lw=2, linestyle='-', label=f'Tahmin μ = {mu_median:.2f}')
ax.set_xlabel("Parlaklık")
ax.set_ylabel("Frekans")
ax.set_title("Sentetik Gözlem Verisi Dağılımı")
ax.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/grafik1_veri_dagilimi.png", dpi=150)
plt.show()

fig2, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
chain = sampler.get_chain()
labels = [r"$\mu$ (Parlaklık)", r"$\sigma$ (Hata)"]
for i, (axi, label) in enumerate(zip(axes, labels)):
    axi.plot(chain[:, :, i], alpha=0.3, lw=0.5, color='steelblue')
    axi.axhline([true_mu, true_sigma][i], color='red', lw=1.5, linestyle='--')
    axi.set_ylabel(label)
axes[-1].set_xlabel("MCMC Adımı")
axes[0].set_title("MCMC Zincir İzleme (Trace Plot)")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/grafik2_trace_plot.png", dpi=150)
plt.show()

fig3 = corner.corner(
    flat_samples,
    labels=[r"$\mu$ (Parlaklık)", r"$\sigma$ (Hata)"],
    truths=[true_mu, true_sigma],
    truth_color='red',
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt=".3f",
    color='steelblue'
)
fig3.suptitle("Corner Plot: Posterior Dağılımı", y=1.02, fontsize=14)
plt.savefig("/mnt/user-data/outputs/grafik3_corner_plot.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n[ÇIKTI] Tüm grafikler kaydedildi.")
print("\n  grafik1_veri_dagilimi.png")
print("  grafik2_trace_plot.png")
print("  grafik3_corner_plot.png")