import re
import math
import random
from collections import Counter

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = text.lower()
text = re.sub(r'[^\w\s]', '', text)
words = text.split()

split_index = int(0.8 * len(words))
train_words = words[:split_index]
test_words = words[split_index:]

def get_ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

unigrams = get_ngrams(train_words, 1)
bigrams = get_ngrams(train_words, 2)
trigrams = get_ngrams(train_words, 3)

unigram_freq = Counter(unigrams)
bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)

vocab_size = len(unigram_freq)
total_unigrams = sum(unigram_freq.values())

def unigram_prob(w):
    return (unigram_freq[(w,)] + 1) / (total_unigrams + vocab_size)

def bigram_prob(w1, w2):
    return (bigram_freq[(w1, w2)] + 1) / (unigram_freq[(w1,)] + vocab_size)

def trigram_prob(w1, w2, w3):
    return (trigram_freq[(w1, w2, w3)] + 1) / (bigram_freq[(w1, w2)] + vocab_size)

def predict_next_word(word):
    candidates = {}
    for w in unigram_freq:
        w2 = w[0]
        candidates[w2] = bigram_prob(word, w2)
    return max(candidates, key=candidates.get)

def generate_text(w1, w2, length=20):
    result = [w1, w2]
    for _ in range(length):
        candidates = {}
        for w in unigram_freq:
            w3 = w[0]
            candidates[w3] = trigram_prob(result[-2], result[-1], w3)
        next_word = max(candidates, key=candidates.get)
        result.append(next_word)
    return " ".join(result)

def perplexity_unigram(test):
    log_prob = 0
    N = len(test)
    for w in test:
        log_prob += math.log(unigram_prob(w))
    return math.exp(-log_prob / N)

def perplexity_bigram(test):
    log_prob = 0
    N = len(test)
    for i in range(1, N):
        log_prob += math.log(bigram_prob(test[i-1], test[i]))
    return math.exp(-log_prob / N)

def perplexity_trigram(test):
    log_prob = 0
    N = len(test)
    for i in range(2, N):
        log_prob += math.log(trigram_prob(test[i-2], test[i-1], test[i]))
    return math.exp(-log_prob / N)

print("Top 10 unigrams:", unigram_freq.most_common(10))
print("Top 10 bigrams:", bigram_freq.most_common(10))
print("Top 10 trigrams:", trigram_freq.most_common(10))

print("Prediction for 'the':", predict_next_word("the"))
print("Generated text:", generate_text(train_words[0], train_words[1], 15))

print("Unigram Perplexity:", perplexity_unigram(test_words))
print("Bigram Perplexity:", perplexity_bigram(test_words))
print("Trigram Perplexity:", perplexity_trigram(test_words))
