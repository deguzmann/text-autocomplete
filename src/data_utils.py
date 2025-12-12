# Инициализация библиотек
import re

# Очистка текстов
def clean_text(text):
    # Удаление ссылок
    text = re.sub(r'http\S+', '', text)

    # Удаление упоминаний
    text = re.sub(r'@\w+', '', text)

    # Удаление лишних пробелов, символов
    text = re.sub(r'\s+', ' ', text).strip() 
    text = re.sub(r'\.(?:\s*\.){1,}', ' ', text)
    text = re.sub(r'!(?:\s*!){1,}', '!', text)
    text = re.sub(r'^-', '', text)

    # к нижнему регистру
    text = text.lower()
    return text
