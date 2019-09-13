import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


def group_sentences(data: pd.DataFrame, column_name: str, group_by="sentence_idx"):
    sentences = data.groupby(group_by).apply(lambda row: " ".join(row[column_name]))
    return sentences.tolist()


def tokenize(sentences: list, to_lower=True):
    tokenizer = Tokenizer(filters='', lower=to_lower)
    tokenizer.fit_on_texts(sentences)
    return tokenizer.texts_to_sequences(sentences), tokenizer


def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')


def split_sentences(sentences: list, piece_length: int):
    res = []

    for sentence in sentences:
        length = len(sentence)
        if length < piece_length:
            res.extend(pad_sequences([sentence], piece_length, padding='post'))
            continue

        res.extend([sentence[x:x + piece_length] for x in range(0, length - piece_length + 1)])

    return np.array(res)


def split_text_sentences(sentences: list, piece_length: int):
    res = []
    for sentence in sentences:
        words = sentence.split(" ")
        length = len(words)

        if length < piece_length:
            res.append(words + [""] * (piece_length - length))
            continue

        res.extend([words[x:x + piece_length] for x in range(0, length - piece_length + 1)])

    return np.array(res)


# def split_text_sentences(sentences: list, piece_length: int):
#     res = []
#     for sentence in sentences:
#         words = sentence.split(" ")
#         length = len(words)
#
#         if length < piece_length:
#             res.append(sentence)
#             continue
#
#         res.extend([" ".join(words[x:x + piece_length]) for x in range(0, length - piece_length + 1)])
#
#     return np.array(res)


def main():
    df = pd.read_csv("data/ner.csv", encoding="ISO-8859-1", error_bad_lines=False)
    df = df.iloc[281835:]
    data = df.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
                    'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
                    'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
                    'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
                    'prev-prev-word', 'prev-shape', 'prev-word', "pos", "shape"], axis=1)
    group_sentences(data, "word")


if __name__ == "__main__":
    main()
