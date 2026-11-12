# cbow_embeddings

A continuous bag-of-words (CBOW) word embedding model in PyTorch, with nearest-neighbour search, analogy tests and a PCA plot.

## What it does

`cbow_embeddings.py` trains word vectors on a small corpus that is written directly in the script: about 100 text-to-image prompts, roughly 2,700 words.

- **Vocabulary:** the 39 most frequent words plus an `<UNK>` token for everything else
- **Training pairs:** the 2 words on each side of a word are the context, the word itself is the target
- **Model:** an embedding layer (20 dimensions), the average of the 4 context vectors, and a linear layer over the vocabulary
- **Training:** cross-entropy loss, Adam with learning rate 0.01, 100 epochs, loss printed every 20 epochs

After training it finds nearest neighbours by cosine similarity, runs analogy tests (`a : b = c : ?`, including a simple gender bias probe) and draws the vectors in 2D with PCA.

## Run

Install the packages from the repository root with `pip install -r requirements.txt`, then:

```
cd cbow_embeddings
python cbow_embeddings.py
```

## Notes

- The words used in the demo queries (`dog`, `prague`, `man`, `woman`, `king`, `czechia`, `paris`, `doctor`) are not among the 39 words of this small vocabulary, so they all map to `<UNK>` and the neighbours and analogies printed for them are not meaningful. To get useful results, query words that are in the vocabulary, or train on a larger corpus and raise `VOCAB_SIZE`.
- Training runs one example at a time, which is simple but slow on a large corpus.
