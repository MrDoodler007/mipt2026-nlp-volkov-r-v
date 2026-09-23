"""Сравнение классического (Yandex Translate) и нейронного перевода.

Текст переводится дважды: ru -> en -> ru. Качество обратного перевода
оценивается средними по предложениям BLEU и METEOR.
Перевод выполняется вручную, результаты сохраняются в текстовые файлы.
"""

import argparse
import re
from pathlib import Path

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from razdel import sentenize

REFERENCE_FILE = Path("reference.txt")
ENGLISH_FILE = Path("yandex_en.txt")
YANDEX_FILE = Path("yandex_ru_roundtrip.txt")
NEURAL_FILE = Path("neural_ru_roundtrip.txt")
RESULTS_FILE = Path("results.txt")
BLEU_SMOOTHING = SmoothingFunction().method2


class ExactWordStemmer:
    def stem(self, word):
        return word


class EmptyWordNet:
    def synsets(self, word):
        return []


METEOR_STEMMER = ExactWordStemmer()
METEOR_WORDNET = EmptyWordNet()


def read_lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path, lines):
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_reference(input_file):
    # razdel корректно разбивает русскую пунктуацию на предложения
    text = input_file.read_text(encoding="utf-8")
    sentences = [sentence.text.strip() for sentence in sentenize(text) if sentence.text.strip()]
    write_lines(REFERENCE_FILE, sentences)
    print(f"Предложений сохранено в {REFERENCE_FILE}: {len(sentences)}")


def tokenize(text):
    return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)


def bleu(reference, hypothesis):
    # BLEU одного предложения только по словам, для n-грамм 1-4
    if not reference or not hypothesis:
        return 0.0
    return sentence_bleu(
        [reference],
        hypothesis,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=BLEU_SMOOTHING,
    )


def meteor(reference, hypothesis):
    # METEOR только по точным совпадениям слов, без стемминга и синонимов
    if not reference or not hypothesis:
        return 0.0
    return meteor_score(
        [reference],
        hypothesis,
        stemmer=METEOR_STEMMER,
        wordnet=METEOR_WORDNET,
    )


def average_scores(reference, hypotheses):
    # Средние BLEU и METEOR по предложениям
    bleus = []
    meteors = []
    for reference_sentence, hypothesis_sentence in zip(reference, hypotheses):
        reference_tokens = tokenize(reference_sentence)
        hypothesis_tokens = tokenize(hypothesis_sentence)
        bleus.append(bleu(reference_tokens, hypothesis_tokens))
        meteors.append(meteor(reference_tokens, hypothesis_tokens))
    return sum(bleus) / len(bleus), sum(meteors) / len(meteors)


def evaluate():
    reference = read_lines(REFERENCE_FILE)
    yandex = read_lines(YANDEX_FILE)
    neural = read_lines(NEURAL_FILE)
    if not reference:
        raise ValueError(f"В файле {REFERENCE_FILE} нет предложений")
    if len(reference) != len(yandex) or len(reference) != len(neural):
        raise ValueError(
            "Число предложений должно совпадать: "
            f"reference={len(reference)}, yandex={len(yandex)}, neural={len(neural)}"
        )

    yandex_bleu, yandex_meteor = average_scores(reference, yandex)
    neural_bleu, neural_meteor = average_scores(reference, neural)

    report = (
        f"Предложений: {len(reference)}\n"
        "Сравнение: исходный русский текст и текст после двойного перевода\n\n"
        "Классический перевод Yandex (ru -> en -> ru)\n"
        f"Средний BLEU: {yandex_bleu:.6f}\n"
        f"Средний METEOR: {yandex_meteor:.6f}\n\n"
        "Нейронный перевод (ru -> en -> ru)\n"
        "Нейросеть: DeepSeek V4 Flash (high reasoning)\n"
        f"Средний BLEU: {neural_bleu:.6f}\n"
        f"Средний METEOR: {neural_meteor:.6f}\n"
    )
    RESULTS_FILE.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Результаты сохранены в {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="Оценка двойного перевода ru -> en -> ru: средние BLEU и METEOR"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Разбить текст на предложения")
    prepare.add_argument("input", type=Path, help="Исходный русский текст (.txt)")
    subparsers.add_parser("evaluate", help="Посчитать средние BLEU и METEOR")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_reference(args.input)
    else:
        evaluate()


if __name__ == "__main__":
    main()
