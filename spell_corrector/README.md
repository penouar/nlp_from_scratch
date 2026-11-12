# spell_corrector

A word-level spelling corrector that uses edit distance, with two ways of finding candidates so their speed can be compared.

## What it does

`spell_corrector.py` builds a dictionary from a small corpus (the `corpus` string at the top of the file) and estimates each word's probability from its frequency. A misspelled word is corrected in one of two ways:

- **Candidate generation:** produce every string at edit distance 1 (deletions, transpositions, replacements, insertions), keep the ones that are known words, and fall back to distance 2 if there are none. The most frequent candidate wins.
- **Brute force:** compute the Levenshtein distance from the word to every dictionary word, keep the closest ones (maximum distance 2) and pick the most frequent among them.

One-letter words are left unchanged, and a word with no candidate is returned as it is.

The script prints Levenshtein distances for a few word pairs, how many strings each method generates for some misspellings, the corrected version of a test sentence with both methods, and a timing table.

## Run

```
cd spell_corrector
python spell_corrector.py
```

Standard library only.

## Sample output

```
LEVENSHTEIN DISTANCE TESTS
kitten       sitting      3
book         back         2
restorant    restaurant   2

GENERATED VARIANT COUNTS
restorant    distance_1 =    494   distance_2 =   114324
tonigth      distance_1 =    390   distance_2 =    70184

CORRECTION BY CANDIDATE GENERATION
i will go to the restaurant tonight and order a delicious dessert before going home
```

The input sentence was `I well go to the restorant tonigth and order a delicshus desert befor going home`.

## Notes

- The built-in dictionary is tiny, so scanning every word is faster than generating tens of thousands of candidate strings. Generation does not depend on dictionary size, so the balance changes with a larger corpus.
- Corrections can only be as good as the corpus. Replace the `corpus` text with something bigger for real use.
- Timings depend on the machine.
