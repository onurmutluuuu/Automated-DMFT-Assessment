# YOLOv11 ile Panoramik Görüntülerde Otomatik DMFT Değerlendirmesi

## Proje Özeti
Bu proje, diş radyolojisinde Decayed (Çürük), Missing (Eksik) ve Filled (Dolgulu) dişlerin (DMFT indeksi) otomatik olarak değerlendirilmesi amacıyla geliştirilmiş derin öğrenme tabanlı bir sistemdir. Çalışma kapsamında en güncel YOLOv11 konvolüsyonel sinir ağı (CNN) mimarisi kullanılarak, hem klinik karar verme süreçlerini destekleyecek hem de epidemiyolojik çalışmalara yön verecek yüksek doğruluklu bir araç tasarlanmıştır.

## Öne Çıkan Özellikler
* **YOLOv11 Entegrasyonu:** Diş tespiti ve sınıflandırılması için gelişmiş YOLOv11 nesne tespiti algoritması kullanıldı.
* **Eksik Diş Çıkarımı (Inference):** Eksik dişler, doğrudan radyograf üzerinden eğitilmek yerine, modelin tespit ettiği dişlerin 32 dişlik tam referans çerçevesi ile algoritmik olarak karşılaştırılmasıyla belirlenmektedir.
* **Klinik Web Arayüzü:** Hekimlerin radyografları sisteme yükleyip interaktif diş şemaları üzerinden DMFT profillerini anında inceleyebilmeleri için web uygulaması geliştirildi.

## Proje Yapısı ve İş Akışı
Proje dizini, makalede belirtilen metodolojik adımları modüler bir şekilde takip edecek biçimde yapılandırılmıştır:

* `DATA/`: Radyografik görüntülerin ve YOLO formatındaki `TXT` etiket dosyalarının bulunduğu ana veri klasörü.
* `GUI/`: Vue.js, FastAPI ve MSSQL entegrasyonu ile geliştirilen klinik web uygulamasının kaynak kodlarını içerir.
* `Data_Split_for_Training_Testing_and_Vali...py`: 1316 adet görüntüyü eğitim (n=1053), doğrulama (n=132) ve test (n=131) setlerine rastgele ayıran betik.
* `step1_preprocessing.py`: Görüntülerin modele uygun hale getirilmesi için 1280x1280 çözünürlüğe ölçeklenmesi (downsampling) ve piksel yoğunluklarının normalize edilmesi.
* `step2_augmentation.py`: Kontrast ve pozisyon farklılıklarını tolere edebilmek için eğitim verisine rastgele parlaklık (%±44) ve gürültü (%4) ekleyen veri artırma (augmentation) modülü.
* `step3_training.py`: PyTorch framework'ü ve SGD optimizasyon algoritması kullanılarak YOLOv11 modelinin eğitilmesi.
* `step4_evaluation.py`: Modelin test verisi üzerindeki Precision, Recall, F1-Skoru ve mAP metriklerinin hesaplanması; hata matrislerinin (confusion matrix) oluşturulması.
* `step5_dmft_calculation.py`: Tespit edilen diş listesinin 32 tam dişlik referans listesiyle karşılaştırılarak tespit edilemeyenlerin 'Eksik (Missing)' olarak işaretlenmesi ve nihai DMFT skorunun algoritmik hesaplanması.
* `run.py`: Projenin modüllerini veya klinik arayüzü başlatan ana yürütme (execution) dosyası.

## Model Performansı
YOLOv11 modeli genel DMFT değerlendirmesinde yüksek bir performans sergilemiş, özellikle dolgulu dişlerin tespitinde öne çıkmıştır. Modelin eksik dişleri algılama doğruluğu **%95.48** olarak ölçülmüştür.

| Metrik | Sağlıklı | Çürük | Dolgulu | Genel |
| :--- | :--- | :--- | :--- | :--- |
| **Hassasiyet (Precision)** | 0.805 | 0.794 | 0.883 | 0.828 |
| **Duyarlılık (Recall)** | 0.899 | 0.597 | 0.937 | 0.811 |
| **F1-Skoru** | 0.849 | 0.682 | 0.909 | 0.819 |
| **mAP@0.5** | 0.863 | 0.699 | 0.954 | 0.839 |

## Teknoloji Yığını (Tech Stack)
* **Derin Öğrenme:** Python (v3.11.10), PyTorch (v2.5.1+cu121), Ultralytics YOLOv11
* **Arayüz (Frontend):** Vue.js
* **Sunucu (Backend):** FastAPI
* **Veritabanı:** MSSQL

## Eğitim Donanımı
Modelin eğitimi ve optimizasyonu aşağıdaki sistem özelliklerine sahip bir iş istasyonunda gerçekleştirilmiştir:
* **GPU:** NVIDIA GeForce RTX 3090
* **CPU:** AMD Ryzen 9 5900x
* **RAM:** 64 GB DDR4 3600MHz

## Yazarlar
* **Elif Aslan** - İzmir Tınaztepe Üniversitesi
* **Ali Canberk Ulusoy** - Sigma Dental Clinic
* **Onur Mutlu** - Karadeniz Teknik Üniversitesi, Bilgisayar Bilimleri Bölümü
* **Erinç Önem** - Ege Üniversitesi
* **Elif Şener** - Ege Üniversitesi
* **Ali Mert** - İzmir Katip Çelebi Üniversitesi
* **B. Güniz Baksi** - Ege Üniversitesi
