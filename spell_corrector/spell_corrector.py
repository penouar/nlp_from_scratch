import re
import time
from collections import Counter

alphabet = 'abcdefghijklmnopqrstuvwxyz'

corpus = """
the restaurant was busy during the evening and many people came for dinner
he decided to order food and stay until late at night
after dinner they enjoyed a sweet dessert and talked for a while
before leaving the place they paid and went home
she will visit the restaurant again because the food was delicious
many customers prefer to order dessert after eating dinner
the chef prepared a delicious meal using fresh ingredients
people often go out for dinner when they want to relax
before going home they stopped to eat something small
the restaurant offers a variety of dishes including dessert
he will travel tonight and return home tomorrow morning
students sometimes go home late after studying all day
a delicious dessert can make the evening better
they plan to order food from a nearby restaurant tonight
good spelling correction depends on word frequency and edit distance
automatic systems can detect and fix small typing mistakes
the probability of a word can be estimated from a text corpus
edit distance is useful for comparing two similar words
people often write messages quickly and make small spelling errors
before dinner they were walking around the city center
"""

def tokenize(text):
    return re.findall(r'[a-z]+', text.lower())

def build_dictionary(text):
    return Counter(tokenize(text))

dictionary = build_dictionary(corpus)
total_words = sum(dictionary.values())

def probability(word):
    return dictionary[word] / total_words if total_words else 0

def levenshtein_distance(a, b):
    m = len(a)
    n = len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[m][n]

def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [l + r[1:] for l, r in splits if r]
    transposes = [l + r[1] + r[0] + r[2:] for l, r in splits if len(r) > 1]
    replaces = [l + c + r[1:] for l, r in splits if r for c in alphabet]
    inserts = [l + c + r for l, r in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))

def known(candidates):
    return {word for word in candidates if word in dictionary}

def candidate_words(word):
    if word in dictionary:
        return {word}
    return (
        known(edits1(word)) or
        known(edits2(word)) or
        {word}
    )

def correct_word(word):
    return max(candidate_words(word), key=probability)

def brute_force_correct_word(word, max_distance=2):
    if word in dictionary:
        return word

    best_words = []
    best_distance = float('inf')

    for dict_word in dictionary:
        d = levenshtein_distance(word, dict_word)
        if d < best_distance:
            best_distance = d
            best_words = [dict_word]
        elif d == best_distance:
            best_words.append(dict_word)

    if best_distance > max_distance:
        return word

    return max(best_words, key=probability)

def correct_sentence(sentence, method='generate'):
    result = []
    for word in tokenize(sentence):
        if len(word) == 1:
            result.append(word)
        elif method == 'generate':
            result.append(correct_word(word))
        else:
            result.append(brute_force_correct_word(word))
    return ' '.join(result)

def generated_variant_counts(word):
    e1 = edits1(word)
    e2 = edits2(word)
    return len(e1), len(e2)

test_pairs = [
    ('kitten', 'sitting'),
    ('book', 'back'),
    ('flaw', 'lawn'),
    ('restorant', 'restaurant'),
    ('tonigth', 'tonight'),
    ('delicshus', 'delicious')
]

print('LEVENSHTEIN DISTANCE TESTS')
print('-' * 40)
for a, b in test_pairs:
    print(f'{a:12} {b:12} {levenshtein_distance(a, b)}')

print()
print('GENERATED VARIANT COUNTS')
print('-' * 50)
variant_words = ['restorant', 'tonigth', 'delicshus', 'desert', 'befor']
for word in variant_words:
    c1, c2 = generated_variant_counts(word)
    print(f'{word:12} distance_1 = {c1:6}   distance_2 = {c2:8}')

print()
sentence = 'I well go to the restorant tonigth and order a delicshus desert befor going home'

print('CORRECTION BY CANDIDATE GENERATION')
print('-' * 50)
print(correct_sentence(sentence, method='generate'))

print()
print('CORRECTION BY BRUTE-FORCE DICTIONARY SEARCH')
print('-' * 50)
print(correct_sentence(sentence, method='brute'))

print()
print('TIME COMPARISON TABLE')
print('-' * 90)
print(f'{"Word":12} {"Generate Result":20} {"Generate Time (s)":18} {"Brute Result":20} {"Brute Time (s)":15}')
print('-' * 90)

for word in variant_words:
    start = time.perf_counter()
    result1 = correct_word(word)
    time1 = time.perf_counter() - start

    start = time.perf_counter()
    result2 = brute_force_correct_word(word)
    time2 = time.perf_counter() - start

    print(f'{word:12} {result1:20} {time1:<18.6f} {result2:20} {time2:<15.6f}')
