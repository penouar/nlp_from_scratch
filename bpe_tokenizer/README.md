# bpe_tokenizer

Byte-pair encoding (BPE) implemented from scratch and trained in two ways, to compare how the two tokenizers split the same sentence.

## What it does

`bpe_tokenizer.py` reads `data.txt`, lowercases it and normalises the whitespace, then prints the number of words, the number of distinct words and the 20 most frequent words. It trains BPE twice, with 500 merge steps each:

- **Word-level:** every distinct word is split into characters plus an `<end_of_word>` marker, and counted with its frequency in the text.
- **Text-level:** the whole text is treated as one long sequence of characters. The code calls this "byte-level", but it works on characters.

Both are used to tokenise the test sentence `i am an erasmus student from morocco`. The script prints the tokens, the tokens per word, the tokens per 1000 characters and the share of text-level tokens that contain a space.

## Run

Put a plain text file named `data.txt` in this folder, then:

```
cd bpe_tokenizer
python bpe_tokenizer.py
```

Standard library only.

## Notes

- Text-level training works on one huge string, so it gets slow as the file grows. Keep `data.txt` small (tens of KB runs in a few seconds).
- Whitespace is dropped when the text is split into symbols for training, so no learned merge contains a space. The last number printed therefore only counts tokens that are a lone space.
