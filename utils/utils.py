import string, unicodedata, re
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def build_vocab(data):
    vocab = []
    for sentence in data:
        sentence_tokenized = word_tokenize(sentence)
        for word in sentence_tokenized:
            if word not in vocab:
                vocab.append(word)

    return vocab


def build_sequences(texts, word2idx):
    sequences = []
    for sentence in texts:
        sequence = []
        sentence_tokenized = word_tokenize(sentence)
        for token in sentence_tokenized:
            if token in word2idx.keys():
                sequence.append(word2idx[token])
            else:
                sequence.append(word2idx[unk])
        sequences.append(sequence)
    return sequences
