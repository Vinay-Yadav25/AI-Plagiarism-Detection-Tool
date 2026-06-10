"""
analyzer.py  —  100% local AI detection engine.

Detection stack:
  1. N-gram perplexity   — measures text predictability (low = AI-like)
  2. 20+ linguistic features — burstiness, lexical richness, AI phrase patterns,
     sentence length variance, POS diversity, punctuation, hedging, etc.
  3. Weighted rule-based scorer — combines all signals into a final verdict.
     No internet, no API keys, no heavy model downloads required.
"""

import math
import re
import collections
import string
from typing import Any

import nltk
import numpy as np
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# ── Ensure NLTK data ──────────────────────────────────────────────────────────
for pkg in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_STOPWORDS = set(stopwords.words("english"))

# ─────────────────────────────────────────────────────────────────────────────
# AI HALLMARK PHRASES  (manually curated, evidence-based)
# ─────────────────────────────────────────────────────────────────────────────
AI_PHRASES = [
    # Transitions
    r"\bfurthermore\b", r"\bmoreover\b", r"\bin addition\b", r"\badditionally\b",
    r"\bconsequently\b", r"\btherefore\b", r"\bthus\b", r"\bhence\b",
    r"\bin conclusion\b", r"\bto summarize\b", r"\bto sum up\b",
    r"\bin summary\b", r"\boverall\b", r"\bin essence\b",
    # Hedging
    r"\bit is worth noting\b", r"\bit is important to note\b",
    r"\bit should be noted\b", r"\bit is crucial to\b",
    r"\bone must consider\b", r"\bone can argue\b",
    r"\bit is evident that\b", r"\bit is clear that\b",
    r"\bit is undeniable\b", r"\bit goes without saying\b",
    # Filler openers
    r"\bcertainly\b", r"\bundoubtedly\b", r"\binarguably\b",
    r"\bwithout a doubt\b", r"\bin today's (world|society|era|age)\b",
    r"\bin the (modern|contemporary|current) (world|era|age|society)\b",
    r"\bas (an? )?(ai|language model|llm)\b",
    r"\bof course\b", r"\bneedless to say\b",
    # Robotic conclusions
    r"\bin (light|view) of (the above|this|these)\b",
    r"\btaking everything into (account|consideration)\b",
    r"\ball things considered\b",
    r"\bplays? a (crucial|pivotal|vital|key|significant|important) role\b",
    r"\bdelve into\b", r"\btapestry\b", r"\bunlock(ing)? (the potential|new possibilities)\b",
    r"\bnavigat(e|ing) (the|a) (complex|challenging|dynamic)\b",
]
_AI_PHRASE_RE = [re.compile(p, re.IGNORECASE) for p in AI_PHRASES]


# ─────────────────────────────────────────────────────────────────────────────
# N-GRAM PERPLEXITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class BigramLM:
    """Laplace-smoothed bigram language model."""

    def __init__(self):
        self.unigrams: collections.Counter = collections.Counter()
        self.bigrams: collections.Counter = collections.Counter()
        self.vocab_size = 0

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\b[a-z']+\b", text.lower())
        return tokens

    def train(self, text: str):
        tokens = self._tokenize(text)
        self.unigrams.update(tokens)
        self.bigrams.update(zip(tokens, tokens[1:]))
        self.vocab_size = len(self.unigrams)

    def sentence_perplexity(self, sentence: str) -> float:
        tokens = self._tokenize(sentence)
        if len(tokens) < 2:
            return 500.0
        V = max(self.vocab_size, 1)
        log_prob = 0.0
        for w1, w2 in zip(tokens, tokens[1:]):
            num = self.bigrams[(w1, w2)] + 1          # Laplace smoothing
            den = self.unigrams[w1] + V
            log_prob += math.log(num / den)
        n = len(tokens) - 1
        return math.exp(-log_prob / n)

    def text_perplexity(self, text: str) -> float:
        tokens = self._tokenize(text)
        if len(tokens) < 2:
            return 500.0
        V = max(self.vocab_size, 1)
        log_prob = 0.0
        pairs = list(zip(tokens, tokens[1:]))
        for w1, w2 in pairs:
            num = self.bigrams[(w1, w2)] + 1
            den = self.unigrams[w1] + V
            log_prob += math.log(num / den)
        n = len(pairs)
        return math.exp(-log_prob / n)


# ─────────────────────────────────────────────────────────────────────────────
# LINGUISTIC FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(text: str, lm: BigramLM) -> dict[str, Any]:
    sentences = sent_tokenize(text)
    words_raw = word_tokenize(text)
    words = [w.lower() for w in words_raw if w.isalpha()]
    content_words = [w for w in words if w not in _STOPWORDS]

    n_sents = max(len(sentences), 1)
    n_words = max(len(words), 1)

    # ── 1. Perplexity ──────────────────────────────────────────────────────
    overall_ppl = lm.text_perplexity(text)
    sent_ppls = [lm.sentence_perplexity(s) for s in sentences]
    # Burstiness of perplexity: std dev of sentence perplexities
    ppl_std = float(np.std(sent_ppls)) if len(sent_ppls) > 1 else 0.0

    # ── 2. Sentence length statistics ─────────────────────────────────────
    sent_lengths = [len(word_tokenize(s)) for s in sentences]
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    std_sent_len = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0
    # AI tends to produce uniform sentence lengths (low std)
    cv_sent_len = std_sent_len / max(avg_sent_len, 1)   # coefficient of variation

    # ── 3. Lexical richness ────────────────────────────────────────────────
    # Type-Token Ratio (unique words / total words)
    ttr = len(set(words)) / n_words
    # Hapax legomena ratio (words appearing exactly once)
    freq = collections.Counter(words)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    hapax_ratio = hapax / n_words

    # ── 4. AI phrase density ───────────────────────────────────────────────
    text_lower = text.lower()
    ai_hits = sum(1 for pat in _AI_PHRASE_RE if pat.search(text_lower))
    ai_phrase_density = ai_hits / max(n_words / 100, 1)   # hits per 100 words

    # ── 5. POS diversity ──────────────────────────────────────────────────
    pos_tags = pos_tag(words_raw[:500])   # cap for speed
    pos_counts = collections.Counter(tag for _, tag in pos_tags)
    n_pos = max(sum(pos_counts.values()), 1)
    # Adjective and adverb ratio (AI tends to overuse these)
    adj_adv_ratio = (pos_counts.get("JJ", 0) + pos_counts.get("JJR", 0) +
                     pos_counts.get("JJS", 0) + pos_counts.get("RB", 0) +
                     pos_counts.get("RBR", 0)) / n_pos
    # Verb diversity
    verb_types = {w for w, t in pos_tags if t.startswith("VB")}
    verb_diversity = len(verb_types) / max(n_words / 10, 1)

    # ── 6. Punctuation patterns ────────────────────────────────────────────
    comma_rate = text.count(",") / n_words
    semicolon_rate = text.count(";") / n_words
    em_dash_rate = (text.count("—") + text.count("--")) / n_words
    exclaim_rate = text.count("!") / n_words
    question_rate = text.count("?") / n_sents

    # ── 7. Paragraph structure ─────────────────────────────────────────────
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    n_para = max(len(paragraphs), 1)
    avg_para_sents = n_sents / n_para
    # AI often produces very regular paragraph sizes
    para_sent_counts = []
    for para in paragraphs:
        para_sent_counts.append(len(sent_tokenize(para)))
    cv_para = (float(np.std(para_sent_counts)) / max(float(np.mean(para_sent_counts)), 1)
               if len(para_sent_counts) > 1 else 0.0)

    # ── 8. Contraction & informality ──────────────────────────────────────
    contractions = len(re.findall(r"\b\w+n't\b|\b(I'm|you're|he's|she's|it's|we're|they're|I've|I'd|I'll|won't|can't|don't|didn't|isn't|aren't|wasn't|weren't|couldn't|wouldn't|shouldn't|hasn't|haven't|hadn't)\b", text, re.IGNORECASE))
    contraction_rate = contractions / n_words

    # ── 9. Repetition (unigram redundancy) ────────────────────────────────
    top5_share = sum(c for _, c in freq.most_common(5)) / n_words

    # ── 10. Average word length ────────────────────────────────────────────
    avg_word_len = sum(len(w) for w in words) / n_words

    return {
        # Perplexity
        "overall_ppl": overall_ppl,
        "ppl_std": ppl_std,
        "sent_ppls": sent_ppls,
        # Sentence structure
        "avg_sent_len": avg_sent_len,
        "cv_sent_len": cv_sent_len,
        # Lexical
        "ttr": ttr,
        "hapax_ratio": hapax_ratio,
        "top5_share": top5_share,
        "avg_word_len": avg_word_len,
        # AI phrases
        "ai_hits": ai_hits,
        "ai_phrase_density": ai_phrase_density,
        # POS
        "adj_adv_ratio": adj_adv_ratio,
        "verb_diversity": verb_diversity,
        # Punctuation
        "comma_rate": comma_rate,
        "semicolon_rate": semicolon_rate,
        "em_dash_rate": em_dash_rate,
        "exclaim_rate": exclaim_rate,
        "question_rate": question_rate,
        # Structure
        "cv_para": cv_para,
        # Informality
        "contraction_rate": contraction_rate,
        # Counts
        "n_sentences": n_sents,
        "n_words": n_words,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTED RULE-BASED SCORER
# ─────────────────────────────────────────────────────────────────────────────

def score_features(feat: dict) -> tuple[float, list[dict]]:
    """
    Returns (ai_probability 0.0–1.0, list of signal dicts).
    Each signal has: name, observation, direction (ai|human), weight (high|medium|low), contribution.
    """
    signals = []
    score = 0.0   # cumulative weighted AI score
    total_weight = 0.0

    def add(name, observation, direction, weight_label, raw_contribution):
        nonlocal score, total_weight
        w = {"high": 3.0, "medium": 2.0, "low": 1.0}[weight_label]
        contrib = raw_contribution * w   # positive = AI, negative = human
        score += contrib
        total_weight += w
        signals.append({
            "signal": name,
            "observation": observation,
            "direction": direction,
            "weight": weight_label,
            "contribution": round(contrib, 3),
        })

    # ── Perplexity (most reliable signal) ─────────────────────────────────
    ppl = feat["overall_ppl"]
    if ppl < 30:
        add("Perplexity", f"Very low ({ppl:.1f}) — extremely predictable word choices",
            "ai", "high", +0.9)
    elif ppl < 60:
        add("Perplexity", f"Low ({ppl:.1f}) — moderately predictable, typical of AI",
            "ai", "high", +0.65)
    elif ppl < 120:
        add("Perplexity", f"Moderate ({ppl:.1f}) — borderline predictability",
            "mixed", "high", +0.1)
    elif ppl < 250:
        add("Perplexity", f"Elevated ({ppl:.1f}) — varied word choices, lean human",
            "human", "high", -0.4)
    else:
        add("Perplexity", f"High ({ppl:.1f}) — unpredictable, strongly human-like",
            "human", "high", -0.75)

    # ── Perplexity variance (burstiness) ──────────────────────────────────
    pstd = feat["ppl_std"]
    if pstd < 15:
        add("Perplexity Uniformity",
            f"Very consistent across sentences (std={pstd:.1f}) — AI-like evenness",
            "ai", "medium", +0.6)
    elif pstd < 50:
        add("Perplexity Uniformity",
            f"Moderate variation between sentences (std={pstd:.1f})",
            "mixed", "medium", +0.1)
    else:
        add("Perplexity Uniformity",
            f"High variation (std={pstd:.1f}) — human writing is naturally bursty",
            "human", "medium", -0.5)

    # ── Sentence length uniformity ─────────────────────────────────────────
    cv = feat["cv_sent_len"]
    if cv < 0.15:
        add("Sentence Length Variance",
            f"Very uniform lengths (CV={cv:.2f}) — AI typically produces metronomic sentences",
            "ai", "medium", +0.7)
    elif cv < 0.30:
        add("Sentence Length Variance",
            f"Moderate length variation (CV={cv:.2f})",
            "mixed", "medium", +0.1)
    else:
        add("Sentence Length Variance",
            f"High length variation (CV={cv:.2f}) — humans mix short punchy sentences with long ones",
            "human", "medium", -0.5)

    # ── Lexical diversity (TTR) ────────────────────────────────────────────
    ttr = feat["ttr"]
    if ttr > 0.75:
        add("Lexical Diversity (TTR)",
            f"High ({ttr:.2f}) — rich vocabulary, consistent with human writing",
            "human", "medium", -0.4)
    elif ttr > 0.55:
        add("Lexical Diversity (TTR)",
            f"Average ({ttr:.2f}) — typical range",
            "mixed", "low", 0.0)
    else:
        add("Lexical Diversity (TTR)",
            f"Low ({ttr:.2f}) — repetitive vocabulary, AI-like",
            "ai", "medium", +0.45)

    # ── AI phrase density ──────────────────────────────────────────────────
    hits = feat["ai_hits"]
    density = feat["ai_phrase_density"]
    if hits >= 5:
        add("AI Hallmark Phrases",
            f"{hits} detected (density {density:.1f}/100 words) — heavy use of AI-typical transitions and filler",
            "ai", "high", +0.85)
    elif hits >= 2:
        add("AI Hallmark Phrases",
            f"{hits} detected — some AI-typical phrases present",
            "ai", "medium", +0.45)
    elif hits == 1:
        add("AI Hallmark Phrases",
            f"1 detected — minimal AI-typical phrasing",
            "mixed", "low", +0.1)
    else:
        add("AI Hallmark Phrases",
            "None detected — no common AI transitional phrases found",
            "human", "medium", -0.3)

    # ── Contractions / informality ─────────────────────────────────────────
    cr = feat["contraction_rate"]
    if cr > 0.015:
        add("Contractions & Informality",
            f"Contraction rate {cr*100:.1f}% — informal voice suggests human authorship",
            "human", "medium", -0.45)
    elif cr > 0.005:
        add("Contractions & Informality",
            f"Low contraction rate ({cr*100:.1f}%) — somewhat formal",
            "mixed", "low", +0.1)
    else:
        add("Contractions & Informality",
            f"Very few contractions ({cr*100:.1f}%) — stiff formal register typical of AI",
            "ai", "medium", +0.4)

    # ── Hapax ratio (unique word variety) ─────────────────────────────────
    hr = feat["hapax_ratio"]
    if hr > 0.55:
        add("Vocabulary Breadth",
            f"{hr*100:.0f}% of words are unique — broad vocabulary",
            "human", "low", -0.3)
    elif hr < 0.35:
        add("Vocabulary Breadth",
            f"Only {hr*100:.0f}% of words unique — recycled vocabulary",
            "ai", "low", +0.3)

    # ── Adjective / adverb overuse ─────────────────────────────────────────
    aar = feat["adj_adv_ratio"]
    if aar > 0.22:
        add("Adjective/Adverb Overuse",
            f"{aar*100:.0f}% of tokens are adj/adv — AI often over-qualifies",
            "ai", "low", +0.25)

    # ── Punctuation signals ────────────────────────────────────────────────
    if feat["em_dash_rate"] > 0.005:
        add("Em Dash Usage",
            f"Em dashes detected — stylistic choice more common in human writing",
            "human", "low", -0.2)
    if feat["exclaim_rate"] > 0.002:
        add("Exclamation Marks",
            f"Exclamation marks present — emotional expressiveness, human indicator",
            "human", "low", -0.2)
    if feat["question_rate"] > 0.1:
        add("Rhetorical Questions",
            f"Questions per sentence: {feat['question_rate']:.2f} — rhetorical style, human indicator",
            "human", "low", -0.15)

    # ── Paragraph structure regularity ─────────────────────────────────────
    if feat["cv_para"] < 0.1 and feat["n_sentences"] > 6:
        add("Paragraph Regularity",
            "Paragraphs are nearly identical in sentence count — robotic structural symmetry",
            "ai", "medium", +0.4)

    # ── Average word length ────────────────────────────────────────────────
    awl = feat["avg_word_len"]
    if awl > 5.5:
        add("Word Length",
            f"Average word length {awl:.1f} chars — elevated vocabulary, AI-like",
            "ai", "low", +0.2)
    elif awl < 4.0:
        add("Word Length",
            f"Average word length {awl:.1f} chars — simple vocabulary, human-like",
            "human", "low", -0.1)

    # Normalize to 0–1 probability
    if total_weight == 0:
        return 0.5, signals

    raw = score / total_weight           # range roughly -1 to +1
    # Sigmoid to 0–1
    ai_prob = 1 / (1 + math.exp(-raw * 4))

    return round(ai_prob, 4), signals


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE-LEVEL AI FLAGGING
# ─────────────────────────────────────────────────────────────────────────────

def flag_sentences(sentences: list[str], sent_ppls: list[float],
                   lm: BigramLM) -> list[dict]:
    """
    Flag individual sentences that are most AI-like based on:
    - Low perplexity
    - AI phrase presence
    """
    flagged = []
    ppl_threshold = min(40.0, float(np.percentile(sent_ppls, 25)) + 5) if sent_ppls else 40.0

    for sent, ppl in zip(sentences, sent_ppls):
        reasons = []
        if ppl < ppl_threshold and ppl < 80:
            reasons.append(f"low perplexity ({ppl:.1f}) — highly predictable phrasing")
        phrase_hits = [pat.pattern for pat in _AI_PHRASE_RE if pat.search(sent.lower())]
        if phrase_hits:
            readable = phrase_hits[0].replace(r"\b", "").replace("(", "").replace(")", "")
            reasons.append(f'contains AI hallmark phrase: "{readable}"')
        if reasons:
            flagged.append({
                "text": sent,
                "perplexity": round(ppl, 2),
                "reasons": reasons,
            })

    return flagged


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def analyze_text(text: str) -> dict:
    """
    Full 100% local analysis pipeline.
    Returns a rich result dict consumed by app.py.
    """
    text = text.strip()
    if not text:
        return {"error": "Empty text"}

    # 1. Train LM on the text itself (self-perplexity for relative measurement)
    lm = BigramLM()
    lm.train(text)

    # 2. Extract all features
    feat = extract_features(text, lm)

    # 3. Score
    ai_prob, signals = score_features(feat)

    # 4. Classification
    if ai_prob > 0.72:
        classification = "AI Generated"
    elif ai_prob > 0.45:
        classification = "Mixed / Uncertain"
    else:
        classification = "Human Written"

    confidence = int(abs(ai_prob - 0.5) * 2 * 100)   # 0–100, how far from 50/50

    # 5. Sentence-level perplexity breakdown
    sentences = sent_tokenize(text)
    sent_ppls = feat["sent_ppls"]
    sentence_data = [
        {"text": s, "perplexity": round(p, 2)}
        for s, p in zip(sentences, sent_ppls)
    ]

    # 6. Normalize perplexity to human-likeness score (0–100)
    ppl = feat["overall_ppl"]
    clamped = min(max(ppl, 10.0), 600.0)
    human_likeness = round(((clamped - 10) / (600 - 10)) * 100, 1)

    if ppl < 40:
        ppl_interpretation = "Very predictable — strongly AI-like"
    elif ppl < 80:
        ppl_interpretation = "Moderately predictable — likely AI-assisted"
    elif ppl < 150:
        ppl_interpretation = "Moderate variation — borderline"
    elif ppl < 300:
        ppl_interpretation = "Variable — leans human"
    else:
        ppl_interpretation = "Highly varied — consistent with human writing"

    # 7. Flagged sentences
    flagged = flag_sentences(sentences, sent_ppls, lm)

    # 8. Summary text
    top_ai_signals = [s for s in signals if s["direction"] == "ai" and s["weight"] in ("high", "medium")]
    top_human_signals = [s for s in signals if s["direction"] == "human" and s["weight"] in ("high", "medium")]

    if classification == "AI Generated":
        summary = (f"Analysis indicates this text was likely AI-generated with "
                   f"{int(ai_prob*100)}% probability. "
                   + (f"Key indicators: {top_ai_signals[0]['signal'].lower()}, "
                      f"{top_ai_signals[1]['signal'].lower()}." if len(top_ai_signals) >= 2 else ""))
    elif classification == "Human Written":
        summary = (f"Analysis suggests this text is likely human-written with "
                   f"{int((1-ai_prob)*100)}% confidence. "
                   + (f"Key indicators: {top_human_signals[0]['signal'].lower()}." if top_human_signals else ""))
    else:
        summary = ("The text shows mixed signals — it may be AI-assisted, lightly edited AI output, "
                   "or human writing that happens to be very structured.")

    return {
        "classification": classification,
        "confidence": confidence,
        "ai_probability": ai_prob,
        "summary": summary,
        "perplexity": {
            "overall": round(feat["overall_ppl"], 2),
            "normalized": human_likeness,
            "interpretation": ppl_interpretation,
            "sentences": sentence_data,
            "ppl_std": round(feat["ppl_std"], 2),
        },
        "signals": signals,
        "flagged_sentences": flagged,
        "stats": {
            "word_count": feat["n_words"],
            "sentence_count": feat["n_sentences"],
            "char_count": len(text),
            "ttr": round(feat["ttr"], 3),
            "ai_phrases_found": feat["ai_hits"],
            "avg_sentence_length": round(feat["avg_sent_len"], 1),
        },
    }