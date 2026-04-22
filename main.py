# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def plot_rating_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    df["rating"].value_counts().sort_index().plot(kind="bar")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title: str, cmap: str) -> None:
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_wordcloud(text: str, title: str) -> None:
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def predict_sentiment(text: str, vectorizer: TfidfVectorizer, model: LogisticRegression) -> str:
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "Positive review" if pred == 1 else "Negative review"


def main() -> None:
    # Создаём папку для результатов
    os.makedirs("results", exist_ok=True)

    # Загрузка данных
    df = pd.read_csv("data/cleaned_kaspi_reviews.csv")

    # Распределение рейтингов
    plot_rating_distribution(df)

    # Фильтрация русских отзывов и создание метки
    df = df[df["language"] == "russian"].copy()
    df["label"] = (df["rating"] >= 4).astype(int)
    df = df[["combined_text", "label"]].dropna()

    # Деление на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df["combined_text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    # Минимальный список русских стоп-слов
    russian_stopwords = [
        "и", "в", "во", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она",
        "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только",
        "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь",
        "когда", "даже", "ну", "вдруг", "ли", "если", "кстати"
    ]

    # Векторизация текста
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words=russian_stopwords,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # OverSampling только один раз
    ros = RandomOverSampler(random_state=42)
    X_train_over, y_train_over = ros.fit_resample(X_train_vec, y_train)

    # Logistic Regression
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_train_over, y_train_over)

    y_pred_lr = model_lr.predict(X_test_vec)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)

    print("Logistic Regression")
    print(classification_report(y_test, y_pred_lr, target_names=["Negative", "Positive"]))
    print(f"Accuracy: {accuracy_lr:.2%}\n")

    # Naive Bayes
    model_nb = MultinomialNB()
    model_nb.fit(X_train_over, y_train_over)

    y_pred_nb = model_nb.predict(X_test_vec)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)

    print("Naive Bayes")
    print(classification_report(y_test, y_pred_nb, target_names=["Negative", "Positive"]))
    print(f"Accuracy: {accuracy_nb:.2%}\n")

    # Матрицы ошибок
    plot_confusion_matrix(y_test, y_pred_lr, "Confusion Matrix - Logistic Regression", "Blues")
    plot_confusion_matrix(y_test, y_pred_nb, "Confusion Matrix - Naive Bayes", "Greens")

    # Облака слов
    positive_text = " ".join(df[df["label"] == 1]["combined_text"].astype(str))
    negative_text = " ".join(df[df["label"] == 0]["combined_text"].astype(str))

    plot_wordcloud(positive_text, "Positive Reviews Word Cloud")
    plot_wordcloud(negative_text, "Negative Reviews Word Cloud")

    # Топ-термы модели Logistic Regression
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model_lr.coef_[0]

    top_pos = feature_names[np.argsort(coefs)[-20:]][::-1]
    top_neg = feature_names[np.argsort(coefs)[:20]]

    print("Top positive terms:")
    print(top_pos)
    print("\nTop negative terms:")
    print(top_neg)

    # Пример предсказания
    sample_text = "Звук плохой, перестали работать через неделю"
    prediction = predict_sentiment(sample_text, vectorizer, model_lr)
    print(f"\nSample text: {sample_text}")
    print(f"Prediction: {prediction}")

    # Экспорт результатов
    results = X_test.reset_index(drop=True).to_frame(name="text")
    results["true_label"] = y_test.reset_index(drop=True)
    results["predicted"] = model_lr.predict(X_test_vec)
    results["correct"] = results["true_label"] == results["predicted"]

    results.to_csv("results/kaspi_sentiment_predictions.csv", index=False)
    print("\nResults saved to results/kaspi_sentiment_predictions.csv")


if __name__ == "__main__":
    main()