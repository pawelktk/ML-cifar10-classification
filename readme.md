# CIFAR-10 Image Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-2.0-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

Kompleksowy system do klasyfikacji obrazów z datasetu CIFAR-10, zawierający różne architektury modeli CNN oraz interfejs webowy do testowania.


## 🌟 Funkcje

- **5 różnych architektur modeli**:
  - Prosta CNN
  - CNN z Dropout
  - CNN z Batch Normalization
  - Głęboka CNN
  - Transfer Learning z MobileNetV2

- **Inteligentne trenowanie**:
  - Early Stopping
  - Dynamiczny learning rate
  - Konfiguracja specyficzna dla modelu

- **Interfejs webowy**:
  - Upload własnych obrazów
  - Porównanie wielu modeli
  - Top 3 przewidywania

- **Generowanie raportów**:
  - Krzywe uczenia
  - Macierze pomyłek
  - Szczegółowe metryki (accuracy, precision, recall)

## 💻 Instalacja

1. Sklonuj repozytorium:
```bash
https://github.com/pawelktk/ML-cifar10-classification.git
cd ML-cifar10-classification
```

2. Utwórz i aktywuj środowisko wirtualne (opcjonalne):
```bash
python -m venv venv
source venv/bin/activate
```

3. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

## 🚀 Uruchomienie

1. Trenowanie modeli:
```bash
python -m src.train
```

2. Uruchomienie interfejsu webowego:
```bash
python app.py
```

3. Otwórz w przeglądarce:
```
http://localhost:5000
```

4. Monitorowanie treningu (w nowym terminalu):
```bash
tensorboard --logdir=logs/
```


## 🤖 Dostępne modele

| Model | Architektura | Opis |
|-------|-------------|------|
| Simple CNN | 2 warstwy konwolucyjne + dense | Podstawowy model referencyjny |
| Dropout CNN | CNN z warstwami Dropout | Zapobiega przeuczeniu |
| BatchNorm CNN | CNN z Batch Normalization | Stabilizuje proces uczenia |
| Deep CNN | 3 warstwy konwolucyjne | Bardziej złożona architektura |
| MobileNet Transfer | MobileNetV2 + fine-tuning | Wykorzystuje transfer learning |


## 🤖 Użyte LLM

W projekcie został użyty LLM Deepseek V3 w celu doboru parametrów, wygenerowania pliku readme, frontendu we Flasku i szablonu do raportów.