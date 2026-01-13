# Инициализация библиотек
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import urllib.request

import os
import re

def download_with_urllib(url, save_path):
    """
    Скачивание файла с помощью стандартной библиотеки
    """
    try:
        # Создаем директорию
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Скачиваем файл
        urllib.request.urlretrieve(url, save_path)
        
        print(f"Файл сохранен: {save_path}")
        return True
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

# Очистка текстов
def clean_text(text):
    # Удаление ссылок
    text = re.sub(r'http\S+', '', text)

    # Удаление упоминаний
    text = re.sub(r'@\w+', '', text)

    # Удаление лишних пробелов, символов
    text = re.sub(r'\.(?:\s*\.){1,}', ' ', text)
    text = re.sub(r'!(?:\s*!){1,}', '!', text)
    text = re.sub(r'^-', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\s+', ' ', text).strip() 

    # к нижнему регистру
    text = text.lower()
    return text


def save_text(path: str, df: pd.Series):
    # Сохраняем очищенный текст в файл
    if os.path.exists(path):
        print(f"Путь '{path}' существует")
    else:
        print(f"Путь '{path}' не существует.")
        df.to_csv(path, 
                index=False, 
                header=False, 
                sep='\n',  # каждая строка - новый элемент
                encoding='utf-8')
        print(f"Файл '{path}' сохранен.")
    
    return df


def load_local_fiele_txt(path: str):
    with open(path, 'r') as f:
        raw = f.readlines()

    # возвращаем датасет в формате датафрейм
    return pd.DataFrame({'tweet': raw})


def analyze_tweet_lengths(tweets_tokens):
    """Анализируем распределение длин твитов"""
    lengths = [len(t) for t in tweets_tokens]
    print(f"Средняя длина: {np.mean(lengths):.1f}")
    print(f"Максимальная: {max(lengths)}")
    print(f"Минимальная: {min(lengths)}")
    print(f"Медиана: {np.median(lengths):.1f}")


def graph_train(history_train):
    plt.figure(figsize=(14, 5))

    # Потери
    plt.subplot(1, 2, 1)
    plt.plot(history_train["epochs"], history_train["train_loss"], 'b-o')
    plt.title("Потери на обучении")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.grid(True)

    # ROUGE
    plt.subplot(1, 2, 2)
    plt.plot(history_train["epochs"], history_train["val_rouge1"], 'g-o', label='ROUGE-1')
    plt.plot(history_train["epochs"], history_train["val_rouge2"], 'r-o', label='ROUGE-2')
    plt.title("Качество на обучении")
    plt.xlabel("Эпоха")
    plt.ylabel("ROUGE")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"\n🔚 Финальные метрики:")
    print(f"   Loss: {history_train['train_loss'][-1]:.4f}")
    print(f"   ROUGE-1: {history_train['val_rouge1'][-1]:.4f}")
    print(f"   ROUGE-2: {history_train['val_rouge2'][-1]:.4f}")
    
