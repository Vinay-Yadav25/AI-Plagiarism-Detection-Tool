# AI-Plagiarism-Detection-Tool 🛡️

I built this because I was tired of every AI detection tool either requiring an API key, sending your text to some server, or making you download a 3GB model just to check a paragraph. TextGuard runs completely on your machine — no internet after install, no accounts, no keys, nothing phoning home.

It's a Streamlit app that takes text (or a file) and tells you how likely it is to be AI-generated. It uses a perplexity-based language model combined with a bunch of linguistic checks I put together after reading a lot of research on how AI writing actually differs from human writing.

---

## Getting started

Clone the repo, install the dependencies, and run it. That's genuinely all there is to it.

```bash
git clone https://github.com/Vinay-Yadav25/AI-Plagiarism-Detection-Tool.git
cd AI-Plagiarism-Detection-Tool
pip install -r requirements.txt
streamlit run app.py
```

It'll open at `http://localhost:8501`. First run might take a few seconds while NLTK downloads its tokenizer data (~3MB, one time only).

---

## What it actually does

The core idea is perplexity. I train a bigram language model on the text you give it, then measure how "surprised" the model is sentence by sentence. AI-generated text tends to be very predictable — it picks safe, expected words in safe, expected orders. Human writing is messier. People use weird punctuation, write one-sentence paragraphs, throw in a contraction, go on a tangent.

Beyond perplexity, it checks about 9 other signals:

- **Sentence length variance** — AI writes like a metronome. Humans don't.
- **AI hallmark phrases** — "Furthermore", "It is worth noting", "In today's world", "delve into", "plays a crucial role" and about 30 others. These show up constantly in AI output.
- **Contraction rate** — AI almost never writes "don't" when it can write "do not".
- **Lexical richness** — how many unique words vs. total words. AI tends to recycle.
- **Paragraph regularity** — AI structures paragraphs symmetrically. Humans don't think in bullet points.
- **Punctuation patterns** — em dashes, exclamation marks, rhetorical questions. Rare in AI, common in humans.
- **Perplexity burstiness** — even if the overall perplexity looks okay, AI text is *consistently* predictable. Human writing has peaks and valleys.

All of these get weighted and combined into a single AI probability score. The UI shows you the breakdown so you can see exactly what triggered the verdict.

---

## File support

You can paste text directly or upload a file. Supported formats:

**Documents:** `.pdf` `.docx` `.txt`

**Code:** `.py` `.js` `.ts` `.jsx` `.tsx` `.java` `.c` `.cpp` `.cs` `.go` `.rs` `.rb` `.php` `.swift` `.kt`

**Other:** `.html` `.css` `.json` `.yaml` `.sql` `.md` `.sh`

---

## Project structure

```
textguard/
├── app.py              # Streamlit frontend
├── requirements.txt
└── utils/
    ├── __init__.py
    ├── analyzer.py     # Perplexity engine + signal scoring
    └── file_handler.py # Text extraction for each file type
```

---

## Dependencies

Nothing exotic. Just the standard Python NLP stack:

- `streamlit` — UI
- `nltk` — tokenization and POS tagging
- `scikit-learn` — used for TF-IDF in earlier experiments, kept for future use
- `numpy` — stats calculations
- `pdfplumber` — PDF text extraction
- `python-docx` — Word document extraction

---

## Limitations worth knowing

This tool is good at catching obviously AI-generated text — the kind that uses "Furthermore" three times and never once writes a sentence under eight words. It's less reliable on short texts (under ~100 words there just isn't enough signal), heavily edited AI output, and text that was written by a human who happens to write very formally.

It's also not a plagiarism checker in the traditional sense — it doesn't compare your text against a database of sources. It purely looks at *how* the text was written, not *where* the content came from.

Use it as one signal among several, not as a final verdict.

---

## Why no model downloads?

Honestly, most of the transformer-based AI detectors you find online are either overfit to ChatGPT 3.5 output specifically, require you to be online, or are just wrappers around a paid API. The linguistic signals approach is less accurate on edge cases but it's fast, transparent, explainable, and works on an airplane. That tradeoff felt worth it for a local tool.

If you want higher accuracy and don't mind the setup, the `Hello-SimpleAI/chatgpt-detector-roberta` model on HuggingFace is decent — but that's a separate project.

---

## Contributing

Issues and PRs welcome. The signal weights in `analyzer.py` are the most obvious place to experiment if you want to tune accuracy on a specific type of text.