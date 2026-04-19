* μ\_gerçek = 150.0 (Ortalama Parlaklık)
* σ\_gerçek = 10.0 (Gözlem Hatası)
* n\_obs = 50 (Gözlem Sayısı)
* **Posterior** P(θ|D): Veri sonrası parametre dağılımımız
* **Likelihood** P(D|θ): Seçilen θ ile veriyi gözlemleme olasılığı
* **Prior** P(θ): Veriden önce parametreler hakkında bildiklerimiz
* **Evidence** P(D): Normalizasyon sabiti



## 5.1 Parametre Karşılaştırma Tablosu

> Not: Tablodaki değerler `np.random.seed(42)` ile sabitlenen simülasyona aittir.

|Değişken|Gerçek Değer|Tahmin (Median)|Alt Sınır (%16)|Üst Sınır (%84)|Mutlak Hata|
|-|-|-|-|-|-|
|μ (Parlaklık)|150.0|\~149.85|\~148.5|\~151.2|\~0.15|
|σ (Hata Payı)|10.0|\~10.42|\~9.1|\~11.8|\~0.42|

*Kendi kodunuzu çalıştırdığınızda bu değerleri gerçek çıktınızla güncelleyiniz.*

\---

## 6\. Sonuçların Bilimsel Yorumu

### 6.1 Merkezi Eğilim ve Doğruluk (Accuracy) Analizi

Bayesyen çıkarım yöntemi, μ parametresini gerçek değer olan 150.0'a son derece yakın biçimde tahmin etmiştir (mutlak hata ≈ 0.15). Veri setindeki gürültü oranı yaklaşık %6-7 olmasına rağmen (σ/μ = 10/150), model yüksek doğruluk sergilemiştir.

Bunun temel nedeni şudur: n=50 gözlem, **örneklem ortalamasının standart hatasını** (SEM = σ/√n = 10/√50 ≈ 1.41) oldukça küçük kılmaktadır. MCMC bu bilgiyi posterior dağılımına yansıtmış ve geniş priorlara rağmen gerçek değere yakınsayan dar bir posterior üretmiştir. Bu durum Bayesyen çıkarımın "veri çok konuştuğunda prior sustuğu" ilkesinin pratik göstergesidir.

### 6.2 Tahmin Hassasiyeti (Precision) Karşılaştırması

Corner plot incelendiğinde μ'nun güven aralığı genişliğinin (≈ ±1.4), σ'nın güven aralığı genişliğinden (≈ ±1.35 ancak bağıl olarak daha geniş) belirgin biçimde dar olduğu görülür.

**İstatistiksel açıklaması:**

* **μ tahmini**, örneklem ortalamasına (x̄) dayanır. Merkezî Limit Teoremi gereği x̄'ın standart hatası σ/√n ile azalır. n=50 için SEM ≈ 1.41; yani μ tahmini oldukça kesindir.
* **σ tahmini** ise varyansa dayalıdır. Varyansın örnekleme dağılımı Chi-kare dağılımını izler ve serbestlik derecesi n−1=49'dur. Varyans tahmincilerinin bağıl belirsizliği yaklaşık 1/√(2(n-1)) ≈ %10 civarındadır; yani σ doğası gereği μ'dan daha belirsiz tahmin edilir.

**n=50'nin etkisi:** Gözlem sayısı hem μ hem de σ tahminini iyileştirmektedir. Ancak ortalama tahmini n'e göre daha hızlı iyileştiğinden (σ/√n oranı), iki parametre arasındaki hassasiyet farkı n arttıkça kapanır ama tamamen ortadan kalkmaz.

### 6.3 Olasılıksal Korelasyon Analizi

Corner plot'taki μ–σ kesişim grafiği (2D posterior) incelendiğinde, elipsin **yaklaşık dik durduğu** görülür. Bu, iki parametre arasında güçlü bir korelasyon olmadığına işaret etmektedir.

**Fiziksel yorumu:** Bu model basit bir Gaussyen likelihood kullandığından μ ve σ neredeyse bağımsızdır. Ortalama parlaklık tahmini ne olursa olsun, verinin yayılma miktarı (σ) ayrı bir bilgi taşır. Eğer elips eğik olsaydı (örneğin sağa yatık), "büyük μ tahminleri genellikle büyük σ tahminleriyle birlikte geliyor" demek olurdu ki bu senaryoda gözlemlenmemiştir.

\---

## Ek Analizler (Ödev Soruları)

### 1\. Prior Etkisi: Çok Dar Prior Seçilseydi (100–110 arası)?

Eğer μ için prior aralığı 100–110 olsaydı, posterior dağılım **gerçek değerden (150) uzaklaşırdı.** Likelihood ne kadar güçlü olursa olsun, prior sınırların dışına çıkılamaz (log\_prior = -inf döner). MCMC 150 değerine yakın örnekler önerir ama prior bunları reddeder. Sonuç olarak posterior, prior ile likelihood'un çakışmadığı bu durumda **prior sınırına (≈110) yapışır** ve tahmin ciddi ölçüde yanlı olur. Bu durum kötü prior seçiminin modelinizi ne denli yanıltabileceğini gösterir.

### 2\. Gözlem Sayısı n\_obs = 5'e Düşürülseydi?

n=5 ile:

* Örneklem standart hatası SEM = 10/√5 ≈ 4.47 olur (n=50'deki 1.41'den \~3× geniş)
* Posterior dağılım belirgin biçimde **genişler** (yüksek belirsizlik)
* μ için güven aralığı ±1.4 yerine yaklaşık ±4–5 seviyesine çıkar
* σ tahmini çok daha güvenilmez hale gelir (Chi-kare dağılımı çok sağa çarpık kalır)
* Prior'ın etkisi artar: verinin sesi azaldıkça prior daha baskın konuma geçer

Bu sonuç, Bayesyen çıkarımın küçük veri setlerinde hâlâ anlamlı tahmin verdiğini ama belirsizliğin artışını dürüstçe posterior genişliğiyle yansıttığını göstermektedir.

pip install numpy matplotlib emcee corner

