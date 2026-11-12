# ngram_language_model

Unigram, bigram and trigram language models built from scratch on a plain text file, with add-one smoothing and perplexity measured on a held-out part of the text.

## What it does

`ngram_language_model.py` reads `data.txt`, lowercases it, removes punctuation and splits it into words. The first 80% of the words are used for training and the last 20% for testing (no shuffling). From the training words it counts unigrams, bigrams and trigrams, then:

- prints the 10 most frequent n-grams of each size
- predicts the most likely word after "the" with the bigram model
- generates 15 words with the trigram model, starting from the first two training words and always taking the most probable next word
- reports the perplexity of the three models on the test words

Probabilities use add-one (Laplace) smoothing, with the number of distinct training words as the vocabulary size.

## Run

Put a plain text file named `data.txt` in this folder, then:

```
cd ngram_language_model
python ngram_language_model.py
```

Standard library only.

## Notes

- Generation is greedy, so it always picks the most frequent continuation and gives the same text on every run.
- With add-one smoothing on a small corpus, the bigram and trigram models can end up with a worse perplexity than the unigram model, because most of their test n-grams never appeared in training.
