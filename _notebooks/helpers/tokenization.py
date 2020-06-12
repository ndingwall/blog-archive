import json
import os
from typing import List, Tuple, Callable

from colors import color, ansilen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sentencepiece as spm
import matplotlib.ticker as ticker
import requests
import transformers

MWAPIKEY = os.environ["MERRIAM_WEBSTER"]
DATA_LOCATION = os.path.expanduser("~/data/tokenization/")


class FormattedTable:

    def __init__(self, tokenizers, max_word_len=14, max_separators=5):
        self.tokenizers = tokenizers
        self.max_word_len = max_word_len
        self.max_separators = max_separators

    @property
    def col_width(self):
        return self.max_word_len + self.max_separators

    @staticmethod
    def ansicenter(s, width):
        extra_chars = width - ansilen(s)
        left = int(extra_chars / 2)
        right = extra_chars - left
        return " " * left + s + " " * right

    def _format_tokenization(self, tokens: List[str], gold_standard: List[str] = None) -> str:
        cell = ""
        if gold_standard:
            tokens, subwords, separators = subwords_are_morphologically_coherent(tokens, gold_standard)
            assert len(subwords) == len(tokens)
            assert len(separators) == len(tokens) + 1
            for i, (token, sw_valid, sep_valid) in enumerate(zip(tokens, subwords, separators)):
                if i >= 1:
                    cell += color("-", fg='dark green', bg=194, style='bold') \
                        if sep_valid else color("-", fg='dark red', style='bold', bg=224)
                cell += color(token, fg='dark green') if sw_valid else color(token, fg='dark red')
            return cell
        else:
            tokens = clean_tokens(tokens)
            return "-".join(tokens)

    def _header(self, include_gold_standard=True):
        tokenizer_names = [name.center(self.col_width) for name, _ in self.tokenizers]
        header = f"{'Word'.center(self.max_word_len)}"
        if include_gold_standard:
            header += f"{'Gold standard'.center(self.col_width)}"
        header += f"{' '.join(tokenizer_names)}"
        return color(header, bg='light blue', style='bold')

    def _row(self, word: str, gold_standard: List[str] = None):
        cells = list()
        for name, tokenizer in self.tokenizers:
            formatted = self._format_tokenization(tokenize(tokenizer, word), gold_standard)
            cells.append(self.ansicenter(formatted, self.col_width))
        row = f"{color(word.center(self.max_word_len), bg='light blue', style='bold')}"
        if gold_standard:
            guide = self.ansicenter(f"{'-'.join(gold_standard)}", self.col_width)
            row += f"{color(guide, bg='light green', style='bold')}"
        row += f"{' '.join(cells)}"
        return row

    def print_table_from_words(self, *examples: str):
        print(self._header(include_gold_standard=False))
        for example in examples:
            print(self._row(example))

    def print_table_from_gold_standard(self, *examples: List[str]):
        print(self._header())
        for example in examples:
            print(self._row("".join(example), example))

    def lookup_and_print(self, *examples: List[str]):
        print(self._header())
        for example in examples:
            pron = get_pronunciation(example)
            print(self._row("".join(pron), pron))


def get_definition(word):
    url = f'https://www.dictionaryapi.com/api/v3/references/collegiate/json/{word}?key={MWAPIKEY}'
    r = requests.get(url)
    return word, json.loads(r.content.decode())


def get_pronunciation(word):
    try:
        word, defn = get_definition(word)
        return defn[0]['hwi']['hw'].split("*")
    except:
        return "?"


def clean_token(token):
    token = token.replace('Ġ', '')  # Used by GPT-2 to indicate a word-start
    token = token.replace('▁', '')  # Used by sentencepiece to indicate a word-start
    token = token.replace('##', '')   # Used by Bert to indicate an internal word
    return token


def clean_tokens(tokens):
    tokens = [clean_token(token) for token in tokens]
    if tokens[0] == "":
        tokens = tokens[1:]  # E.g. ulm.encode_as_pieces('pictorial') -> ['▁', 'pic', 'torial']
    return tokens


def load_vocab(model_type: str, vocab_size_in_thousands: int) -> List[str]:
    vocab = list()
    path = os.path.join(DATA_LOCATION,
                        "pretrained_lms",
                        f"{model_type.lower()}10M_{vocab_size_in_thousands}k.vocab")
    with open(path, 'r') as fp:
        for line in fp.readlines():
            token, score = line.split("\t")
            vocab.append(token)
    return clean_tokens(vocab)


def load_tokenizer(model_type: str, vocab_size_in_thousands: int) -> spm.SentencePieceProcessor:
    tokmodel = spm.SentencePieceProcessor()
    path = os.path.join(DATA_LOCATION,
                        "pretrained_lms",
                        f"{model_type.lower()}10M_{vocab_size_in_thousands}k.model")
    tokmodel.load(path)
    return tokmodel


def tokenize(tokenizer, s: str) -> List[str]:
    """Standard interface to tokenize either Transformers or SentencePiece tokenizers"""
    if isinstance(tokenizer, transformers.tokenization_gpt2.GPT2Tokenizer):
        s = " " + s  # Need to add prefix space. See https://huggingface.co/transformers/model_doc/gpt2.html#transformers.GPT2TokenizerFast
    if isinstance(tokenizer, transformers.tokenization_utils.PreTrainedTokenizer):
        func = tokenizer.tokenize
    elif isinstance(tokenizer, spm.SentencePieceProcessor):
        func = tokenizer.encode_as_pieces
    else:
        raise TypeError
    return func(s)


def display_tokens(tokens):
    tokens = [clean_token(token) for token in tokens]
    return "-".join(tokens)


def load_gold_standard() -> List[str]:
    df = pd.read_csv(os.path.join(DATA_LOCATION, "merriam_webster_pronunciation_guides.csv"), header=None)
    pronunciations = df[1].values
    examples = [p.split("*") for p in pronunciations if isinstance(p, str)]
    return examples


def plot_subword_growth():
    examples = load_gold_standard()
    unique_subwords = set()
    n_unique_subwords = list()
    for word in examples:
        unique_subwords = unique_subwords.union(word)
        n_unique_subwords.append(len(unique_subwords))
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.lineplot(range(len(n_unique_subwords)), n_unique_subwords, palette='dark')  # , c='b')
    plt.xlabel("Number of words")
    plt.ylabel("Unique subwords")
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    plt.show()


def frac_splits_hit_hyphen_spots(tokenized: List[str], gold_standard: List[str]):
    tokenized = clean_tokens(tokenized)
    if len(tokenized) == 1:
        return 1  # Our tokenizer has the word in its vocabulary.
    token_splits = set(np.cumsum([len(x) for x in tokenized[:-1]]))
    valid_points = set(np.cumsum([len(x) for x in gold_standard[:-1]]))
    return len(token_splits.intersection(valid_points)) / (len(token_splits))


def subwords_are_morphologically_coherent(tokenized: List[str], gold_standard: List[str]):
    tokenized = clean_tokens(tokenized)
    token_splits = [0, *np.cumsum([len(x) for x in tokenized])]
    token_boundaries = [(token_splits[i], token_splits[i + 1]) for i in range(len(tokenized))]
    valid_points = set(np.cumsum([len(x) for x in gold_standard])).union([0])
    subwords = [a in valid_points and b in valid_points for a, b in token_boundaries]
    splits = [boundary in valid_points for boundary in token_splits]
    return tokenized, subwords, splits


def frac_subwords_are_morphologically_coherent(tokenized: List[str], gold_standard: List[str]):
    _, subwords_are_valid, _ = subwords_are_morphologically_coherent(tokenized, gold_standard)
    return np.mean(subwords_are_valid)


def plot_tokenizer_evaluation_scores():
    res = pd.read_csv(os.path.join(DATA_LOCATION, "tokenizer_evaluations.csv"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
    sns.lineplot(x="useful_vocab_size", y="score", hue="model_type",
                 ci=99, data=res, marker='o', ax=ax, err_style='band',
                 markersize=8)
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim([0, 0.7])
    ax.set_xlabel("Relevant vocab size")
    ax.set_ylabel("% morphologically sound subwords")

    bert = res[res['model_type'] == 'Bert'].mean()
    gpt = res[res['model_type'] == 'GPT2'].mean()
    plt.text(bert.useful_vocab_size + 250, bert.score + 0.0125, "Bert", size='large')
    plt.text(gpt.useful_vocab_size + 400, gpt.score - 0.025, "GPT-2", size='large')

    bpe = res[res['model_type'] == 'BPE'].max()
    ulm = res[res['model_type'] == 'ULM'].max()
    bpe = res[(res['model_type'] == 'BPE') & (res['vocab_size'] == 64000)].mean()
    ulm = res[(res['model_type'] == 'ULM') & (res['vocab_size'] == 64000)].mean()
    plt.text(bpe.useful_vocab_size + 750, bpe.score, "BPE", size='large')
    plt.text(ulm.useful_vocab_size + 750, ulm.score, "Unigram LM", size='large')
    ax.legend().set_visible(False)
    plt.show()


def make_thumbnail():
    res = pd.read_csv(os.path.join(DATA_LOCATION, "tokenizer_evaluations.csv"))

    fig, ax = plt.subplots(1, 1, figsize=(4, 5), sharex=True)
    sns.lineplot(x="useful_vocab_size", y="score", hue="model_type",
                 ci=99, data=res, marker='o', ax=ax, err_style='band',
                 markersize=8)
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim([0, 0.7])
    ax.set_xlabel("Relevant vocab size")
    ax.set_ylabel("% morphologically sound subwords")

    bert = res[res['model_type'] == 'Bert'].mean()
    gpt = res[res['model_type'] == 'GPT2'].mean()
    plt.text(bert.useful_vocab_size + 350, bert.score + 0.015, "Bert", size='large')
    plt.text(gpt.useful_vocab_size + 650, gpt.score - 0.035, "GPT-2", size='large')

    bpe = res[res['model_type'] == 'BPE'].max()
    ulm = res[res['model_type'] == 'ULM'].max()
    bpe = res[(res['model_type'] == 'BPE') & (res['vocab_size'] == 64000)].mean()
    ulm = res[(res['model_type'] == 'ULM') & (res['vocab_size'] == 64000)].mean()
    plt.text(bpe.useful_vocab_size - 1000, bpe.score - 0.01, "BPE", size='large', ha='right', rotation=24)
    plt.text(ulm.useful_vocab_size - 5000, ulm.score - 0.07, "Unigram LM", size='large', ha='right', rotation=22)
    ax.legend().set_visible(False)
    plt.savefig("../images/tokenization-preview.png")


def plot_learning_speed():
    learn_times = pd.read_csv(os.path.join(DATA_LOCATION, "time_to_learn.csv"))
    learn_times['time'] = learn_times['time'] / 60
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sns.lineplot(x="vocab_size", y="time", hue="model_type",
                 ci=95, data=learn_times, marker='o', ax=ax,
                 markersize=8)
    plt.xlabel("Vocab size")
    ax.set_ylabel("Time to build (minutes)")
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.show()


def plot_inference_speed():
    res = pd.read_csv(os.path.join(DATA_LOCATION, "tokenizer_evaluations.csv"))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.ylabel("Frac. morphologically sound subwords")
    plt.ylim(0, 120)
    sns.lineplot(x="useful_vocab_size", y="time_per_million", hue="model_type",
                 ci=95, data=res, marker='o', ax=ax)
    plt.xlabel("Relevant vocab size")
    plt.ylabel("Seconds per million tokens")
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.show()
