import re
from pathlib import Path
import numpy as np


def read_and_split(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    sentences_text = re.split(r"[.!?]+", text)
    sentences = []
    for sentence in sentences_text:
        sentence = sentence.lower()
        sentence = re.sub(r"[^а-яё\s]", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if sentence:
            words = sentence.split()
            sentence = []
            for word in words:
                sentence.append(list(word))
            sentences.append(sentence)
    return sentences


# sentences[ sentence[ word[ char ] ] ]
sentences = read_and_split("task 4/мальчики_gt.txt")
