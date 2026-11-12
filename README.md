# nlp-from-scratch

Classic NLP techniques implemented from scratch: language models, spelling correction, subword tokenisation, word embeddings and bilingual word translation. Each folder is a separate, self-contained project.

## Projects

| Folder | What it covers | Needs |
|---|---|---|
| [`ngram_language_model`](ngram_language_model) | Unigram, bigram and trigram models with add-one smoothing, text generation and perplexity | `data.txt` |
| [`spell_corrector`](spell_corrector) | Edit-distance spelling correction: candidate generation against brute-force dictionary search | nothing |
| [`bpe_tokenizer`](bpe_tokenizer) | Byte-pair encoding trained at word level and at text level, compared on one sentence | `data.txt` |
| [`cbow_embeddings`](cbow_embeddings) | CBOW word embeddings in PyTorch with nearest neighbours, analogies and a PCA plot | PyTorch and friends |
| [`cross_lingual_translation`](cross_lingual_translation) | Czech-English word translation with a linear map learned between fastText embedding spaces | NumPy, fastText vectors |

Each folder has its own README with details.

## Setup

Python 3. Only `cbow_embeddings` and `cross_lingual_translation` need packages; the other three use the standard library.

```
pip install -r requirements.txt
```

## Running

Each project runs from inside its own folder:

```
cd spell_corrector
python spell_corrector.py
```

`ngram_language_model` and `bpe_tokenizer` read a plain text file named `data.txt` from their own folder; any text works. Keep it to a few tens of KB for `bpe_tokenizer`, which gets slow on large files. `cross_lingual_translation` needs the fastText vectors described in its README.

## Layout

```
.
├── requirements.txt
├── ngram_language_model/
├── spell_corrector/
├── bpe_tokenizer/
├── cbow_embeddings/
└── cross_lingual_translation/
```
