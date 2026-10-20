# PF replay araçları — particle filter'ı bag üzerinde bilgisayarda çalıştırma

Araçta kaydedilmiş bir bag'deki `/scan` ve odometri mesajlarını **gerçek `particle_filter` koduna**
sırayla verir; PF araçtaymış gibi çalışır. PF'te bir parametreyi / kodu değiştirince sonucunu araca
gitmeden görmek için. Araçtaki PF davranışını birebir üretir (19 Ekim bag'i: tarama tekrar kullanımı,
gecikme, titreme aynı çıktı).

| Dosya | Ne yapar |
|---|---|
| `bag_extract.py` | Bag'deki konular → `.npz` (analiz ve karşılaştırma için) |
| `pf_replay.py` | PF'i bag üzerinde çalıştırır, her güncellemeyi kaydeder (`.npz`) |
| `rp_eval.py` | Bir veya birkaç replay'i tablo halinde karşılaştırır |
| `lagtest.py` | PF pozunun gerçekte kaç ms geç kaldığını gyro'ya göre ölçer (bir viraj aralığında) |

## Kurulum (bir kere, bilgisayarda)
ROS 2 (Humble) ve bu workspace derlenmiş olmalı (`vesc_msgs` bag okumak için lazım).
`range_libc`'yi CPU modunda kur (bilgisayarda CUDA yoksa; araçtaki `rmgpu` yerine replay `rm` kullanır):
```bash
pip3 install --user "cython<3"
git clone https://github.com/f1tenth/range_libc ~/range_libc
cd ~/range_libc/pywrapper && python3 setup.py install --user
python3 -c "import range_libc; print('ok')"
```

## Kullanım
Her terminalde önce:
```bash
source /opt/ros/humble/setup.bash && source ~/RC_Car_Codes/install/setup.bash
T=~/RC_Car_Codes/src/particle_filter/tools
```
1. Bag'i analiz dosyasına çıkar (bag başına bir kere):
   ```bash
   python3 $T/bag_extract.py ~/bags_current/sonbag /tmp/sonbag.npz
   ```
2. PF'i çalıştır — mevcut ayarlarla ve denemek istediğin değişiklikle:
   ```bash
   python3 $T/pf_replay.py ~/bags_current/sonbag /tmp/rp_normal.npz --t1 50
   python3 $T/pf_replay.py ~/bags_current/sonbag /tmp/rp_sync.npz  --t1 50 --set scan_synchronous:=true
   ```
   Seçenekler:
   - `--set ad:=değer ...` : PF parametresi değiştir (ör. `max_particles:=2000 angle_step:=9`)
   - `--config dosya.yaml` : başka config (vars. `config/localize.yaml`; EKF modu için `config/localize_ekf.yaml`)
   - `--map harita.yaml` : başka harita (vars. `maps/my_map__.yaml`)
   - `--t0 / --t1` : bag'in sadece bu saniyeleri (50 s ≈ 20 s sürer)
   - `--init x,y,yaw` : başlangıç pozu (vars. bag'deki ilk `/pf/pose/odom`)
   - `--seed N` : rastgelelik tohumu (sonuç oynuyorsa 2–3 tohumla dene)
3. Karşılaştır:
   ```bash
   python3 $T/rp_eval.py /tmp/sonbag.npz 48 /tmp/rp_normal.npz /tmp/rp_sync.npz
   python3 $T/lagtest.py /tmp/sonbag.npz 26 36 /tmp/rp_normal.npz /tmp/rp_sync.npz
   ```
   `48` = değerlendirilecek son saniye; `26 36` = içinde belirgin bir viraj olan aralık.

PF kodunda değişiklik yaptıysan yeniden derlemen gerekmez: replay `src/particle_filter` altındaki
kaynağı doğrudan kullanır.

## Tablo nasıl okunur (`rp_eval.py`)
| Sütun | Anlamı | İyi olan |
|---|---|---|
| `Hz`, `dt p5-p95` | Güncelleme sıklığı ve aralığın yayılması | Düzenli (dar aralık) |
| `tekrar%` | Aynı lidar taramasını ikinci kez kullanan güncellemeler | 0 |
| `gecikme med/p95` | Tarama damgasından PF çıktısına süre (hesaplama süresi hariç) | Küçük ve sabit |
| `CPU ms` | Güncelleme başına işlem süresi **bu bilgisayarda** (Jetson'da farklı) | — |
| `aracPF fark` | Replay ile araçtaki PF'in konum farkı (replay'in doğruluğunu gösterir) | Küçük |
| `yol/teker` | PF'in gittiği yol / tekerin ölçtüğü yol (düz, hareketli kısımlar) | ~1.0 |
| `durgun adim` | Araç dururken PF'in bir güncellemeden diğerine titremesi | Küçük |
| `ZED-tutarlilik` | Ardışık iki poz arasındaki hareketin ZED odometrisinden farkı | Küçük |

`lagtest.py`: "yayın anı" = poz yayınlandığında ne kadar eski; "yayınlanan damga" = damganın gerçeğe göre hatası (~0 iyi).

## Sınırlar
- Haritanın dışına çıkılan / PF'in kaybolduğu bag parçalarında doğruluk metrikleri anlamsızlaşır; zamanlama metrikleri yine geçerli.
- Hesaplama süresi bilgisayarın CPU'su içindir; Jetson + GPU sürelerini araçta ölç (`top`, `tegrastats`).
