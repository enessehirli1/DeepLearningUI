# Derin Öğrenme Tabanlı Görüntü Sınıflandırma UI

Bu proje, derin öğrenme kullanarak görüntü sınıflandırma işlemlerini gerçekleştiren bir kullanıcı arayüzüdür. Kullanıcılar, çeşitli modeller ve optimizasyon algoritmaları ile eğitim ve test işlemlerini gerçekleştirebilirler. Ayrıca, eğitim sonuçları PostgreSQL veritabanına kaydedilir ve model dosyası olarak kullanıcı bilgisayarına kaydedilebilir.

## Özellikler
* Farklı derin öğrenme modelleri (EffNetB7, ResNet50, ViTB16, EffNetB2) arasında seçim yapma
* Farklı optimizasyon algoritmaları (Adam, SGD) kullanma
* Farklı loss fonksiyonları (Cross-Entropy Loss, NLLoss) arasında seçim yapma
* Eğitim ve test veri setlerini seçme
* Eğitim sürecini izleme ve sonuçları görüntüleme
* Eğitim sonuçlarını ve modeli PostgreSQL veritabanına kaydetme
* Model dosyasını (.pth) kullanıcının bilgisayarına kaydetme

## Kurulum

### Gereksinimler
* Python 3.6 veya üzeri
* PostgreSQL veritabanı

### Adımlar

1. Gerekli bağımlılıkları yükleyin:
  ``` pip install -r requirements.txt ```

2. Veritabanı bağlantı bilgilerini içeren bir .env dosyası oluşturun. Örneğin:
```
DB_NAME=mydatabase
DB_USER=myuser
DB_PASSWORD=mypassword
DB_HOST=localhost
DB_PORT=5432
```

3. .env dosyasını proje kök dizinine yerleştirin ve .gitignore dosyasına ekleyin:


### Veritabanı Tablosu Oluşturma
Aşağıdaki SQL komutunu kullanarak PostgreSQL veritabanınızda gerekli tabloyu oluşturun:
```
CREATE TABLE training_results (
    id SERIAL PRIMARY KEY,
    epoch INT NOT NULL,
    train_loss FLOAT NOT NULL,
    train_acc FLOAT NOT NULL,
    train_precision FLOAT NOT NULL,
    train_recall FLOAT NOT NULL,
    train_f1 FLOAT NOT NULL,
    test_loss FLOAT NOT NULL,
    test_acc FLOAT NOT NULL,
    test_precision FLOAT NOT NULL,
    test_recall FLOAT NOT NULL,
    test_f1 FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Kullanım

1. Projeyi çalıştırmak için:
``` py main.py ```

2. Kullanıcı arayüzü açıldığında, model, optimizer, loss fonksiyonu ve epoch sayısını seçin.

3. Eğitim ve test veri setlerini seçmek için ilgili butonları kullanın.

4. "Model Eğitimine Başla" butonuna tıklayarak eğitim sürecini başlatın.

5. Eğitim süreci tamamlandığında sonuçları görüntüleyin ve kaydetmek için "Kaydet" butonunu kullanın.


## Proje Yapısı

* main.py: Uygulamanın giriş noktası.
* mainWindow.py: Ana pencerenin UI sınıfı.
* mainWindow_functions.py: Ana pencere için işlevler.
* ResultWindow.py: Sonuç penceresinin UI sınıfı.
* ResultWindow_functions.py: Sonuç penceresi için işlevler.
* models.py: Derin öğrenme modellerinin tanımlandığı dosya.
* engine.py: Eğitim ve test işlemlerinin gerçekleştirildiği dosya.
* requirements.txt: Proje bağımlılıklarını listeleyen dosya.
