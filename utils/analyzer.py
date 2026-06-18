"""
analyzer.py  —  TextGuard v3 detection engine.

Key fix: AI-mimicking-human text detection via 6 new targeted signals:
  + Fake informality markers  ("Look,", "Here's the thing", "I'll be honest")
  + Sophisticated vocab in casual context (AI sounds elevated even when casual)
  + Sensory detail overload  (AI emotional writing packs in too many senses)
  + Deliberate rhythmic structure (intentional long-short sentence alternation)
  + I-pronoun sentence ratio  (AI mimics first-person but overuses it uniformly)
  + Lexical sophistication score (MSTTR — moving-window TTR, more robust)
  + Completeness ratio  (AI-mimic writes all grammatically complete sentences)
  + Resolved narrative score  (AI wraps up neatly; humans leave loose ends)

Full 24-signal detection stack — all local, no API keys.
"""

import math, re, collections, string
from typing import Any

import nltk, numpy as np
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tree import Tree

# ── NLTK bootstrap ─────────────────────────────────────────────────────────
for _pkg, _path in [
    ("punkt",                          "tokenizers/punkt"),
    ("punkt_tab",                      "tokenizers/punkt_tab"),
    ("stopwords",                      "corpora/stopwords"),
    ("averaged_perceptron_tagger",     "taggers/averaged_perceptron_tagger"),
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ("maxent_ne_chunker",              "chunkers/maxent_ne_chunker"),
    ("maxent_ne_chunker_tab",          "chunkers/maxent_ne_chunker_tab"),
    ("words",                          "corpora/words"),
]:
    try:    nltk.data.find(_path)
    except LookupError: nltk.download(_pkg, quiet=True)

_STOPWORDS = set(stopwords.words("english"))

# ── AI hallmark phrases (formal writing) ──────────────────────────────────
_FORMAL_AI_PHRASES = [
    r"\bfurthermore\b", r"\bmoreover\b", r"\bin addition\b", r"\badditionally\b",
    r"\bconsequently\b", r"\bhence\b", r"\bin conclusion\b", r"\bto summarize\b",
    r"\bto sum up\b", r"\bin summary\b", r"\bin essence\b", r"\bto conclude\b",
    r"\bin the final analysis\b", r"\bit is worth noting\b",
    r"\bit is important to note\b", r"\bit should be noted\b",
    r"\bit is crucial to\b", r"\bone must consider\b", r"\bone can argue\b",
    r"\bit is evident that\b", r"\bit is clear that\b", r"\bit is undeniable\b",
    r"\bit goes without saying\b", r"\bit is widely (known|acknowledged)\b",
    r"\bof paramount importance\b", r"\bcertainly\b", r"\bundoubtedly\b",
    r"\binarguably\b", r"\bwithout a doubt\b", r"\bneedless to say\b",
    r"\bin today's (world|society|era|age|landscape)\b",
    r"\bin the (modern|contemporary|current|digital) (world|era|age|society)\b",
    r"\bas (an? )?(ai|language model|llm)\b",
    r"\bplays? a (crucial|pivotal|vital|key|significant|important) role\b",
    r"\bdelve into\b", r"\btapestry\b", r"\bunlock(ing)? (the potential|new possibilities)\b",
    r"\bnavigat(e|ing) (the|a) (complex|challenging|dynamic)\b",
    r"\bembark(ing)? on (a|this|the) journey\b",
    r"\bfoster(ing)? (a|an) (culture|environment|sense)\b",
    r"\bseamless(ly)?\b", r"\brobust\b", r"\bcomprehensive\b",
    r"\bsynergy\b", r"\bparadigm shift\b",
    r"\bin order to (ensure|achieve|facilitate|provide)\b",
    r"\bit is (essential|imperative|necessary) (to|that)\b",
    r"\bthe (importance|significance|impact) of\b",
    r"\bsheds? (light|new light) on\b", r"\bpave(s)? the way\b",
    r"\bat the forefront\b",
]

# ── NEW: Fake-informality markers (AI mimicking casual) ───────────────────
_FAKE_INFORMAL_PHRASES = [
    r"\blook,\s", r"\blook —",
    r"\bhere'?s? the thing\b", r"\bhere is the thing\b",
    r"\bi('ll| will) be honest\b", r"\blet me be honest\b",
    r"\bhonestly,\s", r"\btruth (is|be told)\b",
    r"\bthe thing is,?\s", r"\bhere'?s? (what|the) (actually |really )?(worked|happened|matters)\b",
    r"\bsounds obvious,? right\b", r"\brevolutionary,? i know\b",
    r"\bi know,? i know\b", r"\bstay with me\b",
    r"\btrust me (on this|here)\b", r"\byou might be wondering\b",
    r"\blet me (explain|tell you|be clear|break (it|this) down)\b",
    r"\bthe (dirty |ugly |honest )?truth (is|about)\b",
    r"\bhere'?s? (a|the) (secret|thing|reality|catch|twist)\b",
    r"\bplot twist\b", r"\bspoiler alert\b",
    r"\blong story short\b", r"\bcut to the chase\b",
]

# ── NEW: Sophisticated vocab that leaks AI in casual context ──────────────
_ELEVATED_WORDS = re.compile(
    r"\b(drifting|technically|recreate|optimize|obsessing|deliberately|"
    r"precisely|ultimately|fundamentally|essentially|inevitably|simultaneously|"
    r"consequently|intrinsically|paradoxically|poignant|visceral|ephemeral|"
    r"juxtaposition|resonates?|encapsulates?|transcends?|permeates?|"
    r"illuminates?|embodies?|manifests?|underscores?|epitomizes?|"
    r"nuanced|profound|compelling|remarkable|extraordinary|fascinating|"
    r"breathtaking|heartbreaking|bittersweet|inexplicable|undeniable|"
    r"unparalleled|unprecedented|transformative|revolutionary|paradigmatic)\b",
    re.IGNORECASE
)

# ── NEW: Sensory/literary density (AI emotional writing) ─────────────────
_SENSORY_WORDS = re.compile(
    r"\b(smell|scent|aroma|fragrance|taste|flavor|flavour|sound|noise|"
    r"silence|warmth|cold|chill|heat|light|shadow|darkness|texture|"
    r"soft|hard|rough|smooth|bright|dim|golden|glowing|drifting|"
    r"wafting|lingering|fading|echoing|tingling|burning|aching|"
    r"heavy|hollow|empty|full|sharp|gentle|tender|fierce)\b",
    re.IGNORECASE
)

_FORMAL_RE   = [re.compile(p, re.IGNORECASE) for p in _FORMAL_AI_PHRASES]
_INFORMAL_RE = [re.compile(p, re.IGNORECASE) for p in _FAKE_INFORMAL_PHRASES]


# ── Syllable counter ──────────────────────────────────────────────────────
def _syllables(word: str) -> int:
    w = word.lower().strip(".,!?;:\"'")
    if not w: return 0
    c = len(re.findall(r'[aeiou]+', w))
    if w.endswith('e') and c > 1: c -= 1
    return max(1, c)


# ── MSTTR (Moving-window Type-Token Ratio) ────────────────────────────────
def _msttr(words: list, window: int = 50) -> float:
    """More robust lexical diversity — not affected by text length."""
    if len(words) < window:
        return len(set(words)) / max(len(words), 1)
    ttrs = []
    for i in range(0, len(words) - window + 1, window // 2):
        chunk = words[i:i+window]
        ttrs.append(len(set(chunk)) / window)
    return float(np.mean(ttrs)) if ttrs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# N-GRAM LANGUAGE MODEL  (bigram + trigram, Laplace smoothed)
# ─────────────────────────────────────────────────────────────────────────────
class NgramLM:
    def __init__(self, n: int = 2):
        self.n = n
        self.ngrams   = collections.Counter()
        self.contexts = collections.Counter()
        self.vocab    = set()

    def _tok(self, text: str) -> list:
        return re.findall(r"\b[a-z']+\b", text.lower())

    def train(self, text: str):
        t = self._tok(text)
        self.vocab.update(t)
        for i in range(len(t) - self.n + 1):
            ng  = tuple(t[i:i+self.n])
            ctx = ng[:-1]
            self.ngrams[ng]   += 1
            self.contexts[ctx] += 1

    def _log_prob(self, ngram: tuple) -> float:
        V = max(len(self.vocab), 1)
        return math.log((self.ngrams.get(ngram, 0) + 1) /
                        (self.contexts.get(ngram[:-1], 0) + V))

    def perplexity(self, text: str) -> float:
        t = self._tok(text)
        if len(t) < self.n: return 999.0
        pairs = [tuple(t[i:i+self.n]) for i in range(len(t)-self.n+1)]
        lp = sum(self._log_prob(p) for p in pairs)
        return math.exp(-lp / len(pairs))


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (24 features)
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(text: str, bi_lm: NgramLM, tri_lm: NgramLM) -> dict:
    sentences = sent_tokenize(text)
    words_raw = word_tokenize(text)
    words     = [w.lower() for w in words_raw if w.isalpha()]
    n_sents   = max(len(sentences), 1)
    n_words   = max(len(words), 1)
    freq      = collections.Counter(words)
    tl        = text.lower()

    # ── Perplexity ────────────────────────────────────────────────────────
    bi_ppl      = bi_lm.perplexity(text)
    tri_ppl     = tri_lm.perplexity(text)
    overall_ppl = bi_ppl * 0.55 + tri_ppl * 0.45
    sent_ppls   = [bi_lm.perplexity(s) for s in sentences]
    ppl_std     = float(np.std(sent_ppls)) if len(sent_ppls) > 1 else 0.0

    # ── Sentence length stats ─────────────────────────────────────────────
    sent_lens = [len(word_tokenize(s)) for s in sentences]
    avg_sl    = float(np.mean(sent_lens)) if sent_lens else 0.0
    cv_sl     = float(np.std(sent_lens)) / max(avg_sl, 1)
    min_sl    = min(sent_lens) if sent_lens else 0
    max_sl    = max(sent_lens) if sent_lens else 0

    # Fragment ratio — very short sentences (< 5 words): humans write these, AI-mimic rarely does
    fragment_ratio = sum(1 for l in sent_lens if l < 5) / n_sents

    # Completeness ratio — all grammatically "full" sentences (≥ 7 words): high = AI-mimic
    completeness = sum(1 for l in sent_lens if l >= 7) / n_sents

    # ── Rhythmic alternation (deliberate long-short pattern) ──────────────
    # AI-mimic writes deliberately: long sentence. Short one. Long sentence. Short one.
    # Detect intentional alternation vs natural chaos
    alternations = sum(
        1 for i in range(1, len(sent_lens) - 1)
        if (sent_lens[i-1] > 12 and sent_lens[i] < 7) or
           (sent_lens[i-1] < 7  and sent_lens[i] > 12)
    )
    rhythmic_ratio = alternations / max(n_sents - 1, 1)

    # ── Formal AI phrase density ──────────────────────────────────────────
    formal_hits   = sum(1 for p in _FORMAL_RE   if p.search(tl))
    formal_density = formal_hits / max(n_words / 100, 1)

    # ── NEW: Fake-informality density ────────────────────────────────────
    fake_informal_hits    = sum(1 for p in _INFORMAL_RE if p.search(tl))
    fake_informal_density = fake_informal_hits / max(n_words / 100, 1)

    # ── NEW: Elevated vocab in text ───────────────────────────────────────
    elevated_hits = len(_ELEVATED_WORDS.findall(text))
    elevated_density = elevated_hits / max(n_words / 100, 1)

    # ── NEW: Sensory/literary density ────────────────────────────────────
    sensory_hits    = len(_SENSORY_WORDS.findall(text))
    sensory_density = sensory_hits / max(n_words / 100, 1)

    # ── Contraction rate ──────────────────────────────────────────────────
    contractions = len(re.findall(
        r"\b(\w+n't|I'm|you're|he's|she's|it's|we're|they're|"
        r"I've|I'd|I'll|won't|can't|don't|didn't|isn't|aren't|"
        r"wasn't|weren't|couldn't|wouldn't|shouldn't|hasn't|haven't|hadn't)\b",
        text, re.IGNORECASE))
    contraction_rate = contractions / n_words

    # ── MSTTR lexical diversity ───────────────────────────────────────────
    msttr = _msttr(words)
    hapax = sum(1 for c in freq.values() if c == 1)
    hapax_ratio = hapax / n_words

    # ── Shannon entropy ───────────────────────────────────────────────────
    total = sum(freq.values())
    entropy = -sum((c/total)*math.log2(c/total) for c in freq.values() if c > 0)

    # ── Readability ───────────────────────────────────────────────────────
    syllables = sum(_syllables(w) for w in words)
    asl = n_words / n_sents
    asw = syllables / n_words
    flesch = max(0.0, min(100.0, 206.835 - 1.015*asl - 84.6*asw))
    complex_words = sum(1 for w in words if _syllables(w) >= 3)
    fog = 0.4 * (asl + 100 * complex_words / n_words)

    # ── Passive voice ─────────────────────────────────────────────────────
    passive_hits  = len(re.findall(
        r'\b(is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b', text, re.I))
    passive_ratio = passive_hits / n_sents

    # ── Sentence starter diversity ────────────────────────────────────────
    starters = [s.split()[0].lower().strip(string.punctuation)
                for s in sentences if s.split()]
    starter_div   = len(set(starters)) / max(len(starters), 1)
    robotic_words = {"the","this","it","in","furthermore","moreover",
                     "additionally","however","therefore","thus","overall","while"}
    robotic_ratio = sum(1 for s in starters if s in robotic_words) / max(len(starters), 1)

    # NEW: I-pronoun sentence ratio (AI mimic overuses uniform first-person)
    i_ratio = sum(1 for s in starters if s == "i") / max(len(starters), 1)

    # ── Named entity density ──────────────────────────────────────────────
    try:
        pt   = pos_tag(words_raw[:400])
        tree = ne_chunk(pt, binary=True)
        ne_count = sum(1 for chunk in tree if isinstance(chunk, Tree))
    except Exception:
        ne_count = 0
    ne_density = ne_count / max(n_words / 100, 1)

    # ── Punctuation ───────────────────────────────────────────────────────
    em_dash_rate  = (text.count("—") + text.count("--")) / n_words
    exclaim_rate  = text.count("!")  / n_words
    question_rate = text.count("?")  / n_sents
    ellipsis_rate = (text.count("…") + text.count("...")) / n_sents

    # ── Paragraph regularity ──────────────────────────────────────────────
    paras    = [p.strip() for p in text.split("\n\n") if p.strip()]
    para_sc  = [len(sent_tokenize(p)) for p in paras]
    cv_para  = (float(np.std(para_sc)) / max(float(np.mean(para_sc)), 1)
                if len(para_sc) > 1 else 0.0)

    # ── POS diversity ─────────────────────────────────────────────────────
    pos_tags_all = pos_tag(words_raw[:500])
    pos_counts   = collections.Counter(t for _, t in pos_tags_all)
    n_pos        = max(sum(pos_counts.values()), 1)
    adj_adv_ratio = sum(pos_counts.get(t, 0)
                        for t in ("JJ","JJR","JJS","RB","RBR","RBS")) / n_pos
    verb_types   = {w for w, t in pos_tags_all if t.startswith("VB")}
    verb_div     = len(verb_types) / max(n_words / 10, 1)

    # ── N-gram repetition ─────────────────────────────────────────────────
    all_bg    = list(zip(words, words[1:]))
    n_bg      = max(len(all_bg), 1)
    bg_freq   = collections.Counter(all_bg)
    repeat_score = sum(c-1 for c in bg_freq.values() if c > 1) / n_bg

    # ── Internet slang / abbreviations ───────────────────────────────────
    slang_hits = len(re.findall(
        r'\b(lol|lmao|omg|btw|idk|tbh|ngl|rn|imo|fyi|smh|wtf|irl|brb|'
        r'af|fr|nah|yeah|yep|nope|gonna|wanna|gotta|kinda|sorta|ugh|hmm|meh)\b',
        text, re.IGNORECASE))
    # Missing-apostrophe contractions ("its","dont","cant") = casual human typing
    missing_apos = len(re.findall(
        r'\b(its|dont|cant|wont|didnt|isnt|arent|wasnt|werent|couldnt|'
        r'wouldnt|shouldnt|hasnt|havent|hadnt|youre|theyre|theyve|ive|id|im)\b',
        text, re.IGNORECASE))
    # Lowercase sentence starts — AI always capitalises, casual humans often don't
    lowercase_starts = sum(1 for s in sentences if s and s[0].islower())
    lowercase_ratio  = lowercase_starts / n_sents

    # ── Avg word length ───────────────────────────────────────────────────
    avg_word_len = sum(len(w) for w in words) / n_words

    return dict(
        overall_ppl=overall_ppl, bi_ppl=bi_ppl, tri_ppl=tri_ppl,
        ppl_std=ppl_std, sent_ppls=sent_ppls,
        avg_sl=avg_sl, cv_sl=cv_sl, min_sl=min_sl, max_sl=max_sl,
        fragment_ratio=fragment_ratio, completeness=completeness,
        rhythmic_ratio=rhythmic_ratio,
        formal_hits=formal_hits, formal_density=formal_density,
        fake_informal_hits=fake_informal_hits, fake_informal_density=fake_informal_density,
        elevated_hits=elevated_hits, elevated_density=elevated_density,
        sensory_hits=sensory_hits, sensory_density=sensory_density,
        contraction_rate=contraction_rate,
        msttr=msttr, hapax_ratio=hapax_ratio,
        entropy=entropy, flesch=flesch, fog=fog,
        passive_ratio=passive_ratio,
        starter_div=starter_div, robotic_ratio=robotic_ratio, i_ratio=i_ratio,
        ne_density=ne_density,
        em_dash_rate=em_dash_rate, exclaim_rate=exclaim_rate,
        question_rate=question_rate, ellipsis_rate=ellipsis_rate,
        cv_para=cv_para,
        adj_adv_ratio=adj_adv_ratio, verb_div=verb_div,
        repeat_score=repeat_score, avg_word_len=avg_word_len,
        slang_hits=slang_hits, missing_apos=missing_apos,
        lowercase_ratio=lowercase_ratio,
        n_sentences=n_sents, n_words=n_words,
    )


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTED SCORER  (24 signals)
# ─────────────────────────────────────────────────────────────────────────────
def score_features(feat: dict) -> tuple[float, list[dict]]:
    signals      = []
    score        = 0.0
    total_weight = 0.0

    def add(name, obs, direction, wlabel, raw):
        nonlocal score, total_weight
        w = {"high": 3.0, "medium": 2.0, "low": 1.0}[wlabel]
        score        += raw * w
        total_weight += w
        signals.append(dict(signal=name, observation=obs,
                            direction=direction, weight=wlabel,
                            contribution=round(raw*w, 3)))

    ppl = feat["overall_ppl"]

    # ── 1. Perplexity ──────────────────────────────────────────────────────
    if   ppl < 25:  add("Perplexity", f"Extremely low ({ppl:.1f}) — near-deterministic phrasing", "ai",    "high", +0.95)
    elif ppl < 50:  add("Perplexity", f"Low ({ppl:.1f}) — highly predictable, AI-like",           "ai",    "high", +0.70)
    elif ppl < 90:  add("Perplexity", f"Moderate ({ppl:.1f}) — somewhat predictable",              "ai",    "high", +0.25)
    elif ppl < 180: add("Perplexity", f"Elevated ({ppl:.1f}) — varied phrasing, leans human",     "human", "high", -0.35)
    else:           add("Perplexity", f"High ({ppl:.1f}) — unpredictable word choices",            "human", "high", -0.75)

    # ── 2. Trigram coherence ───────────────────────────────────────────────
    if feat["tri_ppl"] < feat["bi_ppl"] * 0.6:
        add("Trigram Coherence",
            f"Trigram PPL ({feat['tri_ppl']:.1f}) much lower than bigram ({feat['bi_ppl']:.1f}) — phrase-level predictability",
            "ai", "medium", +0.50)

    # ── 3. Perplexity burstiness ───────────────────────────────────────────
    pstd = feat["ppl_std"]
    if   pstd < 12: add("Perplexity Uniformity", f"Near-flat σ={pstd:.1f} — robotic consistency",      "ai",    "medium", +0.70)
    elif pstd < 40: add("Perplexity Uniformity", f"Moderate σ={pstd:.1f}",                              "mixed", "medium", +0.10)
    else:           add("Perplexity Uniformity", f"High σ={pstd:.1f} — natural burstiness",             "human", "medium", -0.50)

    # ── 4. Sentence length CV ──────────────────────────────────────────────
    cv = feat["cv_sl"]
    if   cv < 0.12: add("Sentence Length Variance", f"Extremely uniform (CV={cv:.2f})",               "ai",    "medium", +0.80)
    elif cv < 0.28: add("Sentence Length Variance", f"Low variance (CV={cv:.2f})",                    "ai",    "medium", +0.25)
    elif cv < 0.50: add("Sentence Length Variance", f"Normal variance (CV={cv:.2f})",                 "mixed", "low",    +0.05)
    else:           add("Sentence Length Variance", f"High variance (CV={cv:.2f}) — human-like",      "human", "medium", -0.45)

    # ── 5. Fragment ratio (NEW — critical for catching AI mimic) ───────────
    fr = feat["fragment_ratio"]
    if   fr > 0.20: add("Sentence Fragments",
                        f"{fr*100:.0f}% of sentences are fragments (<5 words) — authentic human writing style",
                        "human", "medium", -0.60)
    elif fr > 0.08: add("Sentence Fragments",
                        f"Some fragments ({fr*100:.0f}%) — slight human indicator",
                        "human", "low",    -0.20)
    else:           add("Sentence Fragments",
                        f"No fragments detected — all sentences grammatically complete, AI-mimic pattern",
                        "ai",    "medium", +0.55)

    # ── 6. Completeness ratio (NEW) ────────────────────────────────────────
    comp = feat["completeness"]
    if   comp > 0.95 and feat["n_sentences"] > 5:
        add("Sentence Completeness",
            f"Every sentence is grammatically complete ({comp*100:.0f}%) — AI-mimic writes perfect prose even in casual voice",
            "ai", "medium", +0.50)
    elif comp < 0.70:
        add("Sentence Completeness",
            f"Many incomplete sentences ({(1-comp)*100:.0f}%) — natural human writing",
            "human", "low", -0.25)

    # ── 7. Rhythmic alternation (NEW) ──────────────────────────────────────
    rr = feat["rhythmic_ratio"]
    if   rr > 0.35:
        add("Rhythmic Structure",
            f"Deliberate long-short alternation in {rr*100:.0f}% of transitions — "
            f"AI mimics this pattern intentionally",
            "ai", "medium", +0.45)
    elif rr < 0.10 and feat["n_sentences"] > 6:
        add("Rhythmic Structure",
            "No rhythmic alternation — organic sentence flow",
            "human", "low", -0.10)

    # ── 8. Formal AI phrases ────────────────────────────────────────────────
    fh = feat["formal_hits"]
    if   fh >= 5: add("Formal AI Phrases",    f"{fh} detected — heavy AI transitional language",      "ai",    "high",   +0.90)
    elif fh >= 3: add("Formal AI Phrases",    f"{fh} detected — notable AI phrasing",                "ai",    "high",   +0.60)
    elif fh >= 1: add("Formal AI Phrases",    f"{fh} detected — some AI phrasing",                   "ai",    "low",    +0.20)
    else:         add("Formal AI Phrases",    "None — no formal AI transition language",              "human", "medium", -0.25)

    # ── 9. Fake informality (NEW — key for AI-mimic-casual) ────────────────
    fi = feat["fake_informal_hits"]
    if   fi >= 3: add("Fake Informality Markers",
                      f"{fi} detected ('Look,', 'Here's the thing', 'I'll be honest'…) — "
                      f"AI uses these to simulate casual voice",
                      "ai", "high", +0.85)
    elif fi >= 1: add("Fake Informality Markers",
                      f"{fi} detected — some scripted casual phrases",
                      "ai", "medium", +0.45)

    # ── 10. Elevated vocabulary (NEW — leaks AI in casual context) ──────────
    ed = feat["elevated_density"]
    if   ed > 2.0: add("Elevated Vocabulary",
                       f"{feat['elevated_hits']} sophisticated words in casual context "
                       f"(e.g. 'visceral', 'fundamentally', 'ephemeral') — AI sounds elevated even when informal",
                       "ai", "high", +0.80)
    elif ed > 0.8: add("Elevated Vocabulary",
                       f"{feat['elevated_hits']} elevated words — moderately suspicious in casual writing",
                       "ai", "medium", +0.40)

    # ── 11. Sensory/literary density (NEW — AI emotional writing) ───────────
    sd = feat["sensory_density"]
    if sd > 3.0:
        add("Sensory Language Overload",
            f"{feat['sensory_hits']} sensory words per 100 words — AI packs emotional writing "
            f"with deliberate sensory detail ('warm', 'drifting', 'hollow'…)",
            "ai", "medium", +0.55)
    elif sd > 1.5:
        add("Sensory Language",
            f"Moderate sensory language ({feat['sensory_hits']} hits) — possible AI emotional writing",
            "ai", "low", +0.20)

    # ── 12. Contraction rate ───────────────────────────────────────────────
    cr = feat["contraction_rate"]
    if   cr > 0.025: add("Contractions", f"{cr*100:.1f}% rate — genuine informal voice",             "human", "medium", -0.55)
    elif cr > 0.008: add("Contractions", f"{cr*100:.1f}% — some informality",                        "mixed", "low",    +0.05)
    else:            add("Contractions", f"Very low ({cr*100:.1f}%) — stiff register, AI-like",       "ai",    "medium", +0.45)

    # ── 13. MSTTR lexical diversity ────────────────────────────────────────
    mt = feat["msttr"]
    if   mt > 0.82: add("Lexical Diversity (MSTTR)", f"High ({mt:.3f}) — broad vocabulary",          "human", "medium", -0.35)
    elif mt > 0.68: add("Lexical Diversity (MSTTR)", f"Average ({mt:.3f})",                          "mixed", "low",    +0.00)
    else:           add("Lexical Diversity (MSTTR)", f"Low ({mt:.3f}) — repetitive vocabulary",       "ai",    "medium", +0.40)

    # ── 14. Shannon entropy ────────────────────────────────────────────────
    ent = feat["entropy"]
    if   ent > 7.5: add("Shannon Entropy", f"High ({ent:.2f} bits) — rich information density",      "human", "medium", -0.35)
    elif ent > 6.0: add("Shannon Entropy", f"Moderate ({ent:.2f} bits)",                             "mixed", "low",    +0.00)
    else:           add("Shannon Entropy", f"Low ({ent:.2f} bits) — repetitive vocabulary",           "ai",    "medium", +0.30)

    # ── 15. Flesch readability ─────────────────────────────────────────────
    fl = feat["flesch"]
    if   fl > 80: add("Flesch Ease", f"Very easy ({fl:.0f}) — oversimplified or padded",             "ai",    "low",    +0.20)
    elif 48 <= fl <= 68: add("Flesch Ease", f"Standard band ({fl:.0f}) — AI clusters here",          "ai",    "low",    +0.15)
    elif fl < 30: add("Flesch Ease", f"Very difficult ({fl:.0f}) — dense academic text",              "human", "low",    -0.15)

    # ── 16. Gunning Fog ────────────────────────────────────────────────────
    if feat["fog"] > 16:
        add("Gunning Fog", f"Very high ({feat['fog']:.1f}) — over-complicated phrasing",             "ai",    "low",    +0.20)

    # ── 17. Passive voice ──────────────────────────────────────────────────
    pv = feat["passive_ratio"]
    if   pv > 0.35: add("Passive Voice", f"High ({pv:.2f}/sent) — AI overuses passive",              "ai",    "medium", +0.40)
    elif pv > 0.15: add("Passive Voice", f"Moderate ({pv:.2f}/sent)",                                "mixed", "low",    +0.10)
    else:           add("Passive Voice", f"Low ({pv:.2f}/sent) — active voice, human-like",          "human", "low",    -0.20)

    # ── 18. Sentence starter diversity ────────────────────────────────────
    rob = feat["robotic_ratio"]
    if   rob > 0.55: add("Sentence Openers", f"{rob*100:.0f}% robotic starters",                     "ai",    "medium", +0.55)
    elif rob > 0.30: add("Sentence Openers", f"{rob*100:.0f}% robotic starters — moderate AI signal","ai",    "low",    +0.25)
    else:            add("Sentence Openers", f"Diverse openers (only {rob*100:.0f}% robotic)",        "human", "medium", -0.30)

    # ── 19. I-pronoun uniformity (NEW) ─────────────────────────────────────
    ir = feat["i_ratio"]
    if ir > 0.35:
        add("First-Person Uniformity",
            f"{ir*100:.0f}% of sentences start with 'I' — AI-mimic overuses uniform first-person",
            "ai", "medium", +0.45)

    # ── 20. Named entity density ───────────────────────────────────────────
    ned = feat["ne_density"]
    if   ned > 2.0: add("Named Entities", f"High density ({ned:.1f}/100w) — specific real-world refs", "human", "low", -0.20)
    elif ned < 0.5: add("Named Entities", f"Low density ({ned:.1f}/100w) — vague and general",          "ai",    "low", +0.20)

    # ── 21. Punctuation fingerprint ────────────────────────────────────────
    if feat["em_dash_rate"]  > 0.004: add("Em Dashes",  "Present — deliberate stylistic choice",      "human", "low", -0.20)
    if feat["exclaim_rate"]  > 0.002: add("Exclamations","Present — emotional expressiveness",         "human", "low", -0.20)
    if feat["question_rate"] > 0.10:  add("Questions",  f"Rate {feat['question_rate']:.2f}/sent",      "human", "low", -0.15)
    if feat["ellipsis_rate"] > 0.05:  add("Ellipses",   "Present — hesitant/casual writing style",    "human", "low", -0.15)

    # ── 22. Paragraph regularity ───────────────────────────────────────────
    if feat["cv_para"] < 0.08 and feat["n_sentences"] > 8:
        add("Paragraph Regularity", "Paragraphs nearly identical in length",                          "ai",    "medium", +0.45)
    elif feat["cv_para"] > 0.5:
        add("Paragraph Regularity", "Highly irregular paragraphs — natural human structure",          "human", "low",    -0.20)

    # ── 23. N-gram repetition ──────────────────────────────────────────────
    rs = feat["repeat_score"]
    if   rs > 0.15: add("Phrase Repetition", f"High bigram repetition ({rs:.2f})",                   "ai",    "medium", +0.35)
    elif rs < 0.04: add("Phrase Repetition", f"Low repetition ({rs:.2f}) — varied phrasing",         "human", "low",    -0.15)

    # ── Internet slang / informal markers ─────────────────────────────────
    sh = feat["slang_hits"]
    if   sh >= 2: add("Internet Slang", f"{sh} informal terms (lol, yeah, gonna…) — strong human signal", "human", "high",   -0.75)
    elif sh == 1: add("Internet Slang", "1 informal term — mild human signal",                             "human", "low",    -0.25)

    # ── Apostrophe-less contractions ──────────────────────────────────────
    ma = feat["missing_apos"]
    if   ma >= 2: add("Informal Contractions", f"{ma} words like 'its/dont/cant' — casual human typing",  "human", "medium", -0.60)
    elif ma == 1: add("Informal Contractions", "1 apostrophe-less contraction — minor human indicator",   "human", "low",    -0.20)

    # ── Lowercase sentence starts ─────────────────────────────────────────
    llr = feat["lowercase_ratio"]
    if   llr > 0.50: add("Lowercase Starts", f"{llr*100:.0f}% sentences start lowercase — AI never does this", "human", "high", -0.85)
    elif llr > 0.15: add("Lowercase Starts", f"Some lowercase starts ({llr*100:.0f}%) — informality signal",   "human", "low",  -0.25)

    # ── 24. Adj/adverb overuse ─────────────────────────────────────────────
    aar = feat["adj_adv_ratio"]
    if   aar > 0.22: add("Adj/Adverb Overuse", f"{aar*100:.0f}% adj/adv — AI over-qualifies",        "ai",    "low", +0.25)
    elif aar < 0.10: add("Adj/Adverb Overuse", f"Lean adj/adv ratio ({aar*100:.0f}%)",                "human", "low", -0.10)

    # ── Normalize ──────────────────────────────────────────────────────────
    if total_weight == 0:
        return 0.5, signals
    raw     = score / total_weight
    ai_prob = 1 / (1 + math.exp(-raw * 3.5))
    return round(ai_prob, 4), signals


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE FLAGGING
# ─────────────────────────────────────────────────────────────────────────────
def flag_sentences(sentences, sent_ppls, bi_lm) -> list[dict]:
    flagged   = []
    threshold = min(55.0, float(np.percentile(sent_ppls, 30)) + 10) if sent_ppls else 55.0

    for sent, ppl in zip(sentences, sent_ppls):
        reasons = []
        if ppl < threshold and ppl < 90:
            reasons.append(f"low perplexity ({ppl:.1f}) — predictable phrasing")
        if any(p.search(sent.lower()) for p in _FORMAL_RE):
            reasons.append("formal AI transitional phrase")
        if any(p.search(sent.lower()) for p in _INFORMAL_RE):
            reasons.append("scripted casual phrase (fake informality)")
        if _ELEVATED_WORDS.search(sent):
            words_hit = _ELEVATED_WORDS.findall(sent)
            reasons.append(f"elevated vocab in casual context: {words_hit[0]}")
        if re.search(r'\b(is|are|was|were|be|been)\s+\w+(?:ed|en)\b', sent, re.I):
            reasons.append("passive construction")
        if reasons:
            flagged.append({"text": sent, "perplexity": round(ppl, 2), "reasons": reasons})

    return flagged


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def analyze_text(text: str) -> dict:
    text = text.strip()
    if not text:
        return {"error": "Empty text"}

    bi_lm  = NgramLM(n=2); bi_lm.train(text)
    tri_lm = NgramLM(n=3); tri_lm.train(text)

    feat             = extract_features(text, bi_lm, tri_lm)
    ai_prob, signals = score_features(feat)

    if   ai_prob > 0.68: classification = "AI Generated"
    elif ai_prob > 0.42: classification = "Mixed / Uncertain"
    else:                classification = "Human Written"

    confidence = int(abs(ai_prob - 0.5) * 2 * 100)

    sentences = sent_tokenize(text)
    sent_ppls = feat["sent_ppls"]
    sent_data = [{"text": s, "perplexity": round(p, 2)}
                 for s, p in zip(sentences, sent_ppls)]

    ppl = feat["overall_ppl"]
    clamped    = min(max(ppl, 10.0), 600.0)
    human_like = round(((clamped - 10) / 590) * 100, 1)

    if   ppl < 35:  ppl_interp = "Very predictable — strongly AI-like"
    elif ppl < 70:  ppl_interp = "Moderately predictable — likely AI-assisted"
    elif ppl < 130: ppl_interp = "Borderline — mixed signals"
    elif ppl < 280: ppl_interp = "Variable — leans human"
    else:           ppl_interp = "Highly varied — consistent with human writing"

    flagged    = flag_sentences(sentences, sent_ppls, bi_lm)
    ai_sigs    = [s for s in signals if s["direction"] == "ai"    and s["weight"] in ("high","medium")]
    human_sigs = [s for s in signals if s["direction"] == "human" and s["weight"] in ("high","medium")]

    if   classification == "AI Generated":
        top     = ", ".join(s["signal"].lower() for s in ai_sigs[:2])
        summary = (f"This text was likely generated by an AI ({int(ai_prob*100)}% probability). "
                   + (f"Key indicators: {top}." if top else ""))
    elif classification == "Human Written":
        top     = ", ".join(s["signal"].lower() for s in human_sigs[:2])
        summary = (f"This text reads as human-written ({int((1-ai_prob)*100)}% confidence). "
                   + (f"Key human signals: {top}." if top else ""))
    else:
        summary = ("Mixed signals — this may be AI output edited to sound casual, "
                   "or human writing following a formal/structured template.")

    radar = {
        "Perplexity":    round(1 - min(ppl/400, 1), 2),
        "AI Phrases":    round(min((feat["formal_hits"]+feat["fake_informal_hits"]) / 6, 1), 2),
        "Uniformity":    round(max(0, 1 - feat["cv_sl"] / 0.6), 2),
        "Formality":     round(max(0, 1 - feat["contraction_rate"] / 0.03), 2),
        "Elevated Vocab":round(min(feat["elevated_density"] / 3, 1), 2),
        "Completeness":  round(feat["completeness"], 2),
    }

    return {
        "classification":  classification,
        "confidence":      confidence,
        "ai_probability":  ai_prob,
        "summary":         summary,
        "perplexity": {
            "overall":        round(ppl, 2),
            "bigram":         round(feat["bi_ppl"], 2),
            "trigram":        round(feat["tri_ppl"], 2),
            "normalized":     human_like,
            "interpretation": ppl_interp,
            "sentences":      sent_data,
            "ppl_std":        round(feat["ppl_std"], 2),
        },
        "readability": {
            "flesch":    round(feat["flesch"], 1),
            "fog_index": round(feat["fog"], 1),
            "entropy":   round(feat["entropy"], 2),
        },
        "signals":           signals,
        "flagged_sentences": flagged,
        "radar":             radar,
        "stats": {
            "word_count":          feat["n_words"],
            "sentence_count":      feat["n_sentences"],
            "char_count":          len(text),
            "ttr":                 round(feat["msttr"], 3),
            "ai_phrases_found":    feat["formal_hits"] + feat["fake_informal_hits"],
            "formal_phrases":      feat["formal_hits"],
            "fake_informal":       feat["fake_informal_hits"],
            "elevated_vocab":      feat["elevated_hits"],
            "sensory_words":       feat["sensory_hits"],
            "avg_sentence_len":    round(feat["avg_sl"], 1),
            "fragment_ratio":      round(feat["fragment_ratio"], 2),
            "passive_ratio":       round(feat["passive_ratio"], 2),
            "starter_diversity":   round(feat["starter_div"], 2),
            "slang_hits":           feat["slang_hits"],
            "informal_contractions": feat["missing_apos"],
            "lowercase_starts":     round(feat["lowercase_ratio"], 2),
        },
    }