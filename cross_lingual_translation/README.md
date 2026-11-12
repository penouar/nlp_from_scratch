# cross_lingual_translation

Word translation between Czech and English by learning a linear map between two embedding spaces.

## What it does

`cross_lingual_translation.py` follows the usual embedding-mapping recipe:

1. load the first 50,000 word vectors from the Czech and the English fastText files
2. read a bilingual dictionary and build two aligned matrices `X` (Czech vectors) and `Y` (English vectors) from the pairs where both words are in the loaded vocabulary
3. learn a matrix `W` that minimises `||X W^T - Y||^2` with plain gradient descent (learning rate 0.00005, at most 1000 iterations, stopping early when the loss barely moves over 10 iterations)
4. translate a word by mapping its vector with `W` and returning the 5 English words with the highest cosine similarity
5. report top-1 and top-5 accuracy on the first 500 usable test pairs

The sample word is `pes` (dog).

## Data

The two dictionaries are included. They are Czech-English word pairs, one per line, in the format used by the MUSE bilingual dictionaries (https://github.com/facebookresearch/MUSE):

- `cs-en.train.txt`
- `cs-en.test.txt`

The embeddings are not included because they are very large. Download `cc.cs.300.vec` and `cc.en.300.vec` from https://fasttext.cc/docs/en/crawl-vectors.html, unzip them and put them next to the script. The script reads only the first 50,000 vectors, so you can cut each file down with `head -n 50001` (the first line is a header).

## Run

```
pip install numpy
cd cross_lingual_translation
python cross_lingual_translation.py
```

## Notes

- A source word can have several valid translations in the dictionary. Each pair is scored on its own against a single prediction, so top-1 accuracy cannot reach 100%.
- Lower the learning rate if the loss grows instead of shrinking.
