# Kaspi Reviews Sentiment Analysis

## English

### Overview

This project performs sentiment analysis of product reviews from Kaspi.
The model classifies reviews as positive or negative using machine learning methods.

### Technologies

* Python
* Scikit-learn
* TF-IDF vectorization
* Logistic Regression
* Naive Bayes
* Imbalanced-learn (RandomOverSampler)

### Features

* Text preprocessing for Russian language
* Handling class imbalance using oversampling
* Comparison of two models
* Confusion matrix visualization
* Word cloud generation
* Extraction of important terms

### Results

* Logistic Regression accuracy: 91.78%
* Naive Bayes accuracy: 86.76%

### How to run

```bash id="run1"
pip install -r requirements.txt
python src/main.py
```

### Dataset

The dataset is not included in the repository due to size limitations.

Download it from:
https://drive.google.com/file/d/1V6VTvmqgJl-jHMIYaqBB38FVbx9YhhlX/view

After downloading, place the file in:

```id="path1"
data/cleaned_kaspi_reviews.csv
```

### Example

```python id="ex1"
predict_sentiment("Звук плохой")
# Output: Negative review
```

### Output

After running the script, results will be saved to:
results/kaspi_sentiment_predictions.csv

---

## Русский

### Описание

Проект выполняет анализ тональности отзывов на товары Kaspi.
Модель классифицирует отзывы на позитивные и негативные.

### Технологии

* Python
* Scikit-learn
* TF-IDF
* Logistic Regression
* Naive Bayes
* Oversampling

### Возможности

* Обработка русскоязычных текстов
* Балансировка классов
* Сравнение моделей
* Построение матрицы ошибок
* Облака слов
* Выделение значимых слов

### Результаты

* Logistic Regression: 91.78%
* Naive Bayes: 86.76%

### Запуск

```bash id="run2"
pip install -r requirements.txt
python src/main.py
```

### Данные

Датасет не включен в репозиторий из-за ограничения размера.

Скачать:
https://drive.google.com/file/d/1V6VTvmqgJl-jHMIYaqBB38FVbx9YhhlX/view

После скачивания поместите файл в:

```id="path2"
data/cleaned_kaspi_reviews.csv
```

### Пример

```python id="ex2"
predict_sentiment("Звук плохой")
# Негативный отзыв
```

### Результат

После запуска файл сохраняется в:
results/kaspi_sentiment_predictions.csv
