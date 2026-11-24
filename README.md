# Transformers from scratch: Kazakh poetry and bracket languages

Two labs from the Large Language Models course of my Master's in Computational Linguistics at Université Paris Cité.

## 1. Character-level GPT on Kazakh poetry

`01_char_gpt_kazakh_poetry.ipynb`, report in `report_char_gpt.pdf`.

A decoder-only GPT written in PyTorch and trained on the poems of Magzhan Zhumabaev (`poems.txt`, 571,006 characters, 142 unique characters). Kazakh verse has a regular metre and rhyme, so the question is whether a small character model picks up that structure.

- 861,070 parameters: 4 layers, 128-dimensional embeddings, 4 attention heads, context of 256 characters.
- After 20,000 steps the baseline reached a validation loss of 1.71, or 2.47 bits per character.
- I compared built-in and hand-written causal attention for speed and memory, varied depth and context length, and checked generated text for line length, syllable pattern and rhyme endings.

## 2. Transformers and Dyck languages

`02_dyck_transformer.ipynb`, [open in Colab](https://colab.research.google.com/github/bkadyrbay-hash/llm3lab/blob/main/02_dyck_transformer.ipynb).

A 3-layer BERT-style encoder detects and corrects errors in strings of balanced brackets. This is a clean test of whether a Transformer learns a rule or only patterns from its training distribution.

| Measure | Result |
|---|---|
| Error detection, in distribution | accuracy 1.000 |
| Error detection, out of distribution | accuracy 0.940 |
| Correction, token accuracy | 0.994 |
| Correction, exact match of the full string | 0.515 |

A pushdown automaton baseline stays perfect out of distribution, so the model learned an approximation limited by depth. Token accuracy looks strong only because about half of the tokens need no change, which is why exact match is the number to read. The notebook also covers attention analysis and linear probes for nesting depth.

## Run it

Both notebooks are made for Google Colab with a GPU. The first expects `poems.txt` to be uploaded in its first cell.
