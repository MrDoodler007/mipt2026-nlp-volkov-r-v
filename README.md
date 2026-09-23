# Task 1

1) Распаковать src/lenin_pss_tomy_01-20.rar
2) Запустить скрипт word_embedings.py

# Task 2

Запуск из каталога `task 2-1` / `task 2-2`.

1. Создать виртуальное окружение: `python -m venv ../.venv`
2. Установить зависимости: `../.venv/bin/python -m pip install -r requirements.txt`
3. Подготовить референс: `../.venv/bin/python translation_evaluation.py prepare sample.txt`
4. Перевести текст через переводчик вручную (ru -> en -> ru) и сохранить в `yandex_ru_roundtrip.txt`
5. Сохранить перевод нейросети в `neural_ru_roundtrip.txt`
6. Сравнить: `../.venv/bin/python translation_evaluation.py evaluate`

Результаты:
| Модель | BLEU (книга) | METEOR (книга) | BLEU (статья) | METEOR (статья) |
|---|---|---|---|---|
| Yandex Translate | 0.318 | 0.548 | 0.607 | 0.789 |
| DeepSeek V4 Flash | 0.474 | 0.695 | 0.684 | 0.822 |
