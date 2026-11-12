import re
from collections import Counter

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = text.lower()
text = text.strip()
text = re.sub(r"\s+", " ", text)

words = text.split()
num_words = len(words)
unique_words = len(set(words))
word_freq = Counter(words)

print("Number of words:", num_words)
print("Number of unique words:", unique_words)
print("Top 20 words:")
for word, count in word_freq.most_common(20):
    print(word, count)

def build_word_vocab(word_freq):
    vocab = {}
    for word, freq in word_freq.items():
        chars = " ".join(list(word)) + " <end_of_word>"
        vocab[chars] = freq
    return vocab

def get_pair_frequencies(vocab):
    pair_freq = {}
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_freq[pair] = pair_freq.get(pair, 0) + freq
    return pair_freq

def merge_pair(pair, vocab):
    new_vocab = {}
    pair_str = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(pair_str, replacement)
        new_vocab[new_word] = freq
    return new_vocab

def train_bpe(vocab, K):
    merge_rules = []
    for _ in range(K):
        pair_freq = get_pair_frequencies(vocab)
        if not pair_freq:
            break
        best_pair = max(pair_freq, key=pair_freq.get)
        vocab = merge_pair(best_pair, vocab)
        merge_rules.append(best_pair)
    return vocab, merge_rules

def tokenize_word(word, merge_rules):
    symbols = list(word) + ["<end_of_word>"]
    for pair in merge_rules:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                symbols[i] = symbols[i] + symbols[i + 1]
                del symbols[i + 1]
            else:
                i += 1
    return symbols

def build_byte_vocab(text):
    vocab = {}
    sequence = " ".join(list(text))
    vocab[sequence] = 1
    return vocab

def tokenize_text(text, merge_rules):
    symbols = list(text)
    for pair in merge_rules:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                symbols[i] = symbols[i] + symbols[i + 1]
                del symbols[i + 1]
            else:
                i += 1
    return symbols

word_vocab = build_word_vocab(word_freq)
word_vocab, word_merge_rules = train_bpe(word_vocab, 500)

byte_vocab = build_byte_vocab(text)
byte_vocab, byte_merge_rules = train_bpe(byte_vocab, 500)

test_text = "i am an erasmus student from morocco"

word_tokens = []
for w in test_text.split():
    word_tokens.extend(tokenize_word(w, word_merge_rules))

byte_tokens = tokenize_text(test_text, byte_merge_rules)

print("\nWord-level tokens:", word_tokens)
print("Byte-level tokens:", byte_tokens)

num_words_test = len(test_text.split())
chars_test = len(test_text)

print("Word-level tokens per word:", len(word_tokens) / num_words_test)
print("Byte-level tokens per word:", len(byte_tokens) / num_words_test)
print("Word-level tokens per 1000 chars:", (len(word_tokens) / chars_test) * 1000)
print("Byte-level tokens per 1000 chars:", (len(byte_tokens) / chars_test) * 1000)

cross = sum(1 for t in byte_tokens if " " in t)
print("Byte-level cross-word ratio:", cross / len(byte_tokens))
