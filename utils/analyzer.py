"""
analyzer.py  —  TextGuard v4 detection engine.

Root cause of v3 failures:
  - Perplexity was self-trained (LM trained on same text = circular, always low)
  - Sensory language signal fired on ALL descriptive human writing
  - Rhythmic structure signal fired on good human prose
  - Sentence completeness fired on formal human writing
  - Shannon entropy unreliable on short texts

v4 fixes:
  - Perplexity demoted: kept but heavily downweighted and only as corroboration
  - Removed broken signals: sensory language (standalone), rhythmic structure,
    sentence completeness (standalone), Shannon entropy (standalone)
  - Added reliable new signals: tense uniformity, "we" pronoun indicator,
    specificity (numbers/names), question diversity, epistemic hedge density
  - Thresholds hardened: require stronger evidence to call AI or Human
  - Each signal calibrated on evidence, not assumption
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

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL WORD LISTS  (evidence-based, manually curated)
# ─────────────────────────────────────────────────────────────────────────────

# Formal AI transitions — very high precision for standard AI output
_FORMAL_AI = [
    r"\bfurthermore\b", r"\bmoreover\b", r"\bin addition\b", r"\badditionally\b",
    r"\bconsequently\b", r"\bhence\b", r"\bin conclusion\b", r"\bto summarize\b",
    r"\bto sum up\b", r"\bin summary\b", r"\bin essence\b", r"\bto conclude\b",
    r"\bin the final analysis\b", r"\bit is worth noting\b",
    r"\bit is important to note\b", r"\bit should be noted\b",
    r"\bit is crucial to\b", r"\bone must consider\b", r"\bone can argue\b",
    r"\bit is evident that\b", r"\bit is clear that\b", r"\bit is undeniable\b",
    r"\bit goes without saying\b", r"\bit is widely (known|acknowledged)\b",
    r"\bof paramount importance\b", r"\bundoubtedly\b", r"\binarguably\b",
    r"\bwithout a doubt\b", r"\bneedless to say\b",
    r"\bin today'?s? (world|society|era|age|landscape)\b",
    r"\bin the (modern|contemporary|current|digital) (world|era|age|society)\b",
    r"\bas (an? )?(ai|language model|llm)\b",
    r"\bplays? a (crucial|pivotal|vital|key|significant|important) role\b",
    r"\bdelve into\b", r"\btapestry\b",
    r"\bnavigat(e|ing) (the|a) (complex|challenging|dynamic)\b",
    r"\bembark(ing)? on (a|this|the) journey\b",
    r"\bseamless(ly)?\b", r"\brobust solution\b", r"\bcomprehensive (approach|framework|overview)\b",
    r"\bsynergy\b", r"\bparadigm shift\b",
    r"\bin order to (ensure|achieve|facilitate|provide)\b",
    r"\bit is (essential|imperative|necessary) (to|that)\b",
    r"\bsheds? (light|new light) on\b", r"\bpave(s)? the way\b",
    r"\bat the forefront\b", r"\bthe (importance|significance|impact) of\b",
]

# Fake-casual markers — AI mimicking informal voice
_FAKE_CASUAL = [
    r"\blook,\s", r"\blook —", r"\bhere'?s? the thing\b", r"\bhere is the thing\b",
    r"\bi('ll| will) be honest\b", r"\blet me be honest\b",
    r"\bthe thing is,?\s", r"\bsounds obvious,? right\b",
    r"\brevolutionary,? i know\b", r"\bi know,? i know\b",
    r"\btrust me (on this|here)\b", r"\byou might be wondering\b",
    r"\blet me (explain|tell you|break (it|this) down)\b",
    r"\bthe (dirty |ugly |honest )?truth (is|about)\b",
    r"\bhere'?s? (a|the) (secret|thing|reality|catch|twist)\b",
    r"\bplot twist\b", r"\blong story short\b", r"\bcut to the chase\b",
    r"\bspoiler( alert)?\b",
]

# Elevated vocabulary — sounds polished even in casual context
_ELEVATED = re.compile(
    r"\b(fundamentally|essentially|inevitably|simultaneously|consequently|"
    r"intrinsically|paradoxically|visceral|ephemeral|juxtaposition|"
    r"resonates?\b|encapsulates?\b|transcends?\b|permeates?\b|"
    r"illuminates?\b|embodies?\b|manifests?\b|underscores?\b|epitomizes?\b|"
    r"nuanced|profound(ly)?|compelling(ly)?|unparalleled|unprecedented|"
    r"transformative|paradigmatic|multifaceted|holistic(ally)?|"
    r"substantive(ly)?|meticulous(ly)?|exhaustive(ly)?|seminal)\b",
    re.IGNORECASE
)

# Epistemic hedges (AI overuses these in analytical writing)
_HEDGES = re.compile(
    r"\b(it is (crucial|important|worth|essential|necessary|vital)|"
    r"one must|should be noted|it should|we must consider|it can be argued|"
    r"it is widely|research (suggests|indicates|shows|demonstrates)|"
    r"studies (show|suggest|indicate|have shown)|"
    r"evidence (suggests|indicates|points to)|"
    r"it (appears|seems) (that|to)|arguably|presumably)\b",
    re.IGNORECASE
)

# Real informality — strong human indicators
_SLANG = re.compile(
    r"\b(lol|lmao|omg|btw|idk|tbh|ngl|rn|imo|fyi|smh|wtf|irl|brb|"
    r"af\b|fr\b|nah|yeah|yep|nope|gonna|wanna|gotta|kinda|sorta|"
    r"ugh|hmm|meh|omfg|rofl|ikr|smth|ty\b|np\b)\b",
    re.IGNORECASE
)

# Missing-apostrophe contractions (human casual typing)
_MISSING_APOS = re.compile(
    r"\b(its(?!\s+\w)|dont|cant|wont|didnt|isnt|arent|wasnt|werent|"
    r"couldnt|wouldnt|shouldnt|hasnt|havent|hadnt|youre|theyre|"
    r"theyve|ive|im(?=\s))\b",
    re.IGNORECASE
)

_FORMAL_RE  = [re.compile(p, re.IGNORECASE) for p in _FORMAL_AI]
_CASUAL_RE  = [re.compile(p, re.IGNORECASE) for p in _FAKE_CASUAL]

# ── NEW: Encyclopedic / Wikipedia-style AI patterns ─────────────────────
# AI writing factual content avoids formal transitions but uses these instead
_ENCYCLOPEDIC = [
    r"\bprimarily found in\b", r"\bconsist(s)? of (related|numerous|various|many)\b",
    r"\bare (easily|widely|commonly|generally|often) (recognized|known|found|used|considered)\b",
    r"\bplay(s)? an important role\b", r"\bplay(s)? a (crucial|vital|key|significant) role\b",
    r"\bhave been declining\b", r"\bconservation efforts\b",
    r"\bensure their survival\b", r"\bfor future generations\b",
    r"\bare primarily\b", r"\bmainly (hunt|found|used|consist|feed|live)\b",
    r"\boften referred to as\b", r"\bare known (for|as|to be)\b",
    r"\bthroughout history\b", r"\bthroughout the world\b",
    r"\bbalance of (the|their|an) ecosystem\b",
    r"\bdue to (habitat|climate|human|environmental) (loss|change|conflict|destruction|activity)\b",
    r"\bdespite their (strength|size|power|dominance|intelligence)\b",
    r"\b(these|such|those) (magnificent|remarkable|extraordinary|fascinating|incredible|majestic) (animals|creatures|beings|plants|organisms)\b",
    r"\b(symbol|symbols) of (power|strength|courage|wisdom|freedom|hope|unity)\b",
    r"\bin many cultures (throughout|around|across)\b",
    r"\bare carnivores (and|that|which)\b",
    r"\bhabitat loss\b", r"\bhuman.wildlife conflict\b",
    r"\bare found (in|across|throughout|primarily)\b",
    r"\bsmall population\b", r"\brelated females\b",
    r"\bcontrolling .{3,30} population\b",
    r"\bare (unique|distinct|notable|remarkable) (among|in|for|because)\b",
    r"\b(their|its) (impressive|distinctive|characteristic|remarkable) (mane|features?|adaptations?|abilities)\b",
]
_ENCYCLOPEDIC_RE = [re.compile(p, re.IGNORECASE) for p in _ENCYCLOPEDIC]


# ── Syllable counter ──────────────────────────────────────────────────────
def _syl(word: str) -> int:
    w = word.lower().strip(string.punctuation)
    if not w: return 0
    c = len(re.findall(r'[aeiou]+', w))
    if w.endswith('e') and c > 1: c -= 1
    return max(1, c)


# ── MSTTR ─────────────────────────────────────────────────────────────────
def _msttr(words: list, window: int = 50) -> float:
    if len(words) < window:
        return len(set(words)) / max(len(words), 1)
    ttrs = [len(set(words[i:i+window])) / window
            for i in range(0, len(words) - window + 1, window // 2)]
    return float(np.mean(ttrs)) if ttrs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PERPLEXITY ENGINE  (kept but demoted)
# ─────────────────────────────────────────────────────────────────────────────
class NgramLM:
    def __init__(self, n=2):
        self.n = n
        self.ng  = collections.Counter()
        self.ctx = collections.Counter()
        self.V   = set()

    def _tok(self, t): return re.findall(r"\b[a-z']+\b", t.lower())

    def train(self, text):
        t = self._tok(text)
        self.V.update(t)
        for i in range(len(t) - self.n + 1):
            ng = tuple(t[i:i+self.n])
            self.ng[ng]       += 1
            self.ctx[ng[:-1]] += 1

    def perplexity(self, text):
        t = self._tok(text)
        if len(t) < self.n: return 999.0
        pairs = [tuple(t[i:i+self.n]) for i in range(len(t)-self.n+1)]
        V = max(len(self.V), 1)
        lp = sum(math.log((self.ng.get(p,0)+1)/(self.ctx.get(p[:-1],0)+V))
                 for p in pairs)
        return math.exp(-lp / len(pairs))


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(text: str, bi_lm: NgramLM, tri_lm: NgramLM) -> dict:
    sentences = sent_tokenize(text)
    words_raw = word_tokenize(text)
    words     = [w.lower() for w in words_raw if w.isalpha()]
    n_sents   = max(len(sentences), 1)
    n_words   = max(len(words), 1)
    freq      = collections.Counter(words)
    tl        = text.lower()

    # ── Perplexity (supporting evidence only) ────────────────────────────
    bi_ppl      = bi_lm.perplexity(text)
    tri_ppl     = tri_lm.perplexity(text)
    overall_ppl = bi_ppl * 0.55 + tri_ppl * 0.45
    sent_ppls   = [bi_lm.perplexity(s) for s in sentences]
    ppl_std     = float(np.std(sent_ppls)) if len(sent_ppls) > 1 else 0.0

    # ── Sentence length stats ─────────────────────────────────────────────
    sent_lens      = [len(word_tokenize(s)) for s in sentences]
    avg_sl         = float(np.mean(sent_lens)) if sent_lens else 0.0
    cv_sl          = float(np.std(sent_lens)) / max(avg_sl, 1)
    fragment_ratio = sum(1 for l in sent_lens if l < 5) / n_sents

    # ── Formal AI phrase density ──────────────────────────────────────────
    formal_hits     = sum(1 for p in _FORMAL_RE if p.search(tl))
    fake_casual_hits= sum(1 for p in _CASUAL_RE if p.search(tl))
    all_ai_phrase_hits = formal_hits + fake_casual_hits

    # ── Elevated vocabulary ───────────────────────────────────────────────
    elevated_hits    = len(_ELEVATED.findall(text))
    elevated_density = elevated_hits / max(n_words / 100, 1)

    # ── Epistemic hedges ──────────────────────────────────────────────────
    hedge_hits    = len(_HEDGES.findall(text))
    hedge_density = hedge_hits / max(n_words / 100, 1)

    # ── Genuine informality ───────────────────────────────────────────────
    slang_hits   = len(_SLANG.findall(text))
    missing_apos = len(_MISSING_APOS.findall(text))
    contractions = len(re.findall(
        r"\b(\w+n't|I'm|you're|he's|she's|it's|we're|they're|"
        r"I've|I'd|I'll|won't|can't|don't|didn't|isn't|aren't|"
        r"wasn't|weren't|couldn't|wouldn't|shouldn't|hasn't|haven't|hadn't)\b",
        text, re.IGNORECASE))
    total_informal = contractions + missing_apos
    contraction_rate = total_informal / n_words

    # Lowercase sentence starts
    lowercase_starts = sum(1 for s in sentences if s and s[0].islower())
    lowercase_ratio  = lowercase_starts / n_sents

    # ── NEW: Tense uniformity ─────────────────────────────────────────────
    # AI tends to stay in one tense; humans mix naturally
    pos_tags_all = pos_tag(words_raw[:500])
    past_verbs    = sum(1 for _,t in pos_tags_all if t == 'VBD')
    present_verbs = sum(1 for _,t in pos_tags_all if t in ('VBZ','VBP','VB'))
    total_verbs   = max(past_verbs + present_verbs, 1)
    # 0 = all one tense, 1 = perfectly mixed
    tense_mix = min(past_verbs, present_verbs) / total_verbs

    # ── NEW: "We" pronoun AI indicator ───────────────────────────────────
    # AI analysis/instructional writing uses "we" a lot; personal human writing doesn't
    we_count  = len(re.findall(r'\bwe\b', text, re.I))
    i_count   = len(re.findall(r'\bI\b', text))
    we_ratio  = we_count / n_sents

    # ── NEW: Specificity score ────────────────────────────────────────────
    # Humans include specific numbers, names, dates
    # AI generalises ("many", "several", "various", "numerous")
    specific_numbers = len(re.findall(r'\b\d+(?:\.\d+)?(?:%|st|nd|rd|th)?\b', text))
    vague_quantifiers = len(re.findall(
        r'\b(many|several|various|numerous|countless|myriad|plethora|'
        r'a number of|a variety of|a range of|a wide range|some|certain)\b',
        text, re.IGNORECASE))
    specificity = specific_numbers / max(n_words / 50, 1)
    vagueness   = vague_quantifiers / max(n_words / 50, 1)

    # ── NEW: Question diversity ───────────────────────────────────────────
    questions = [s for s in sentences if s.strip().endswith('?')]
    q_rate    = len(questions) / n_sents

    # ── NEW: Sentence-ending variety ─────────────────────────────────────
    endings        = [s.strip()[-1] for s in sentences if s.strip()]
    ending_variety = len(set(endings)) / max(len(endings), 1)

    # ── Sentence starter analysis ─────────────────────────────────────────
    starters = [s.split()[0].lower().strip(string.punctuation)
                for s in sentences if s.split()]
    robotic_words = {"the","this","it","in","furthermore","moreover",
                     "additionally","however","therefore","thus","overall","while","as"}
    robotic_ratio = sum(1 for s in starters if s in robotic_words) / max(len(starters), 1)
    i_start_ratio = sum(1 for s in starters if s == "i") / max(len(starters), 1)

    # ── Passive voice ─────────────────────────────────────────────────────
    passive_hits  = len(re.findall(r'\b(is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b', text, re.I))
    passive_ratio = passive_hits / n_sents

    # ── Readability ───────────────────────────────────────────────────────
    syllables = sum(_syl(w) for w in words)
    asl = n_words / n_sents
    asw = syllables / n_words
    flesch = max(0.0, min(100.0, 206.835 - 1.015*asl - 84.6*asw))
    complex_w = sum(1 for w in words if _syl(w) >= 3)
    fog = 0.4 * (asl + 100 * complex_w / n_words)

    # ── Lexical diversity ─────────────────────────────────────────────────
    msttr = _msttr(words)

    # ── Named entities ────────────────────────────────────────────────────
    try:
        tree     = ne_chunk(pos_tag(words_raw[:400]), binary=True)
        ne_count = sum(1 for c in tree if isinstance(c, Tree))
    except Exception:
        ne_count = 0
    ne_density = ne_count / max(n_words / 100, 1)

    # ── Punctuation ───────────────────────────────────────────────────────
    em_dash_rate  = (text.count("—") + text.count("--")) / n_words
    exclaim_rate  = text.count("!")  / n_words
    ellipsis_rate = (text.count("…") + text.count("...")) / n_sents

    # ── N-gram repetition ─────────────────────────────────────────────────
    bgs          = list(zip(words, words[1:]))
    bg_freq      = collections.Counter(bgs)
    repeat_score = sum(c-1 for c in bg_freq.values() if c > 1) / max(len(bgs), 1)

    # ── Paragraph regularity ──────────────────────────────────────────────
    paras   = [p.strip() for p in text.split("\n\n") if p.strip()]
    para_sc = [len(sent_tokenize(p)) for p in paras]
    cv_para = (float(np.std(para_sc)) / max(float(np.mean(para_sc)), 1)
               if len(para_sc) > 1 else 0.0)

    # ── NEW: Encyclopedic / Wikipedia-style AI patterns ───────────────────
    encyclopedic_hits    = sum(1 for p in _ENCYCLOPEDIC_RE if p.search(text))
    encyclopedic_density = encyclopedic_hits / max(n_words / 100, 1)

    # ── NEW: Vocabulary sophistication (avg syllables per content word) ───
    avg_syllables = sum(_syl(w) for w in words) / n_words

    # ── NEW: Clause complexity (commas per sentence) ──────────────────────
    comma_per_sent = text.count(',') / n_sents

    # ── NEW: Adj-noun pair density (AI packs in descriptive adjectives) ───
    # Exclude common classifiers that aren't AI-style descriptors
    _NON_DESCRIPTIVE = {"female","male","adult","young","old","human","wild","domestic","local","national","global","social","natural","physical","general","special","common","public","private","large","small","big","little"}
    pos_tags_adj = pos_tag(words_raw[:400])
    adj_noun_pairs = sum(
        1 for i in range(len(pos_tags_adj)-1)
        if pos_tags_adj[i][1].startswith('JJ')
        and pos_tags_adj[i+1][1].startswith('NN')
        and pos_tags_adj[i][0].lower() not in _NON_DESCRIPTIVE
    )
    adj_noun_density = adj_noun_pairs / max(n_words / 20, 1)

    # ── NEW: Repeated subject starter (student/human essay pattern) ───────
    starter_counts = collections.Counter(starters)
    max_starter_repeats = max(starter_counts.values()) if starter_counts else 0
    repeated_starter_ratio = max_starter_repeats / max(n_sents, 1)

    return dict(
        overall_ppl=overall_ppl, bi_ppl=bi_ppl, tri_ppl=tri_ppl,
        ppl_std=ppl_std, sent_ppls=sent_ppls,
        avg_sl=avg_sl, cv_sl=cv_sl, fragment_ratio=fragment_ratio,
        formal_hits=formal_hits, fake_casual_hits=fake_casual_hits,
        all_ai_phrase_hits=all_ai_phrase_hits,
        elevated_hits=elevated_hits, elevated_density=elevated_density,
        hedge_hits=hedge_hits, hedge_density=hedge_density,
        slang_hits=slang_hits, missing_apos=missing_apos,
        contraction_rate=contraction_rate,
        lowercase_ratio=lowercase_ratio,
        tense_mix=tense_mix,
        we_ratio=we_ratio, we_count=we_count, i_count=i_count,
        specificity=specificity, vagueness=vagueness,
        specific_numbers=specific_numbers, vague_quantifiers=vague_quantifiers,
        q_rate=q_rate, ending_variety=ending_variety,
        robotic_ratio=robotic_ratio, i_start_ratio=i_start_ratio,
        passive_ratio=passive_ratio,
        flesch=flesch, fog=fog, msttr=msttr,
        ne_density=ne_density,
        em_dash_rate=em_dash_rate, exclaim_rate=exclaim_rate,
        ellipsis_rate=ellipsis_rate,
        repeat_score=repeat_score, cv_para=cv_para,
        encyclopedic_hits=encyclopedic_hits, encyclopedic_density=encyclopedic_density,
        avg_syllables=avg_syllables, comma_per_sent=comma_per_sent,
        adj_noun_density=adj_noun_density, repeated_starter_ratio=repeated_starter_ratio,
        n_sentences=n_sents, n_words=n_words,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCORER  — evidence-based, calibrated
# ─────────────────────────────────────────────────────────────────────────────
def score_features(feat: dict) -> tuple[float, list[dict]]:
    sigs = []
    score = 0.0
    total_w = 0.0

    def add(name, obs, direction, wlabel, raw):
        nonlocal score, total_w
        w = {"high": 3.0, "medium": 2.0, "low": 1.0}[wlabel]
        score   += raw * w
        total_w += w
        sigs.append(dict(signal=name, observation=obs,
                         direction=direction, weight=wlabel,
                         contribution=round(raw*w, 3)))

    # ════════════════════════════════════════════════════════
    # GROUP A: HIGH-PRECISION AI SIGNALS
    # These almost never appear in genuine human writing
    # ════════════════════════════════════════════════════════

    # A1. Formal AI transition phrases
    fh = feat["formal_hits"]
    if   fh >= 5: add("Formal AI Phrases", f"{fh} formal transitions (furthermore, moreover, in conclusion…)", "ai", "high", +0.90)
    elif fh >= 3: add("Formal AI Phrases", f"{fh} formal transitions detected",                                 "ai", "high", +0.70)
    elif fh >= 1: add("Formal AI Phrases", f"{fh} formal transition phrase(s)",                                 "ai", "medium", +0.40)
    else:         add("Formal AI Phrases", "None — no formal AI transitional language",                         "human", "medium", -0.20)

    # A2. Fake-casual markers
    fc = feat["fake_casual_hits"]
    if   fc >= 3: add("Scripted Casual Voice", f"{fc} fake-casual phrases ('Look,', 'Here's the thing', 'Revolutionary, I know'…)", "ai", "high",   +0.85)
    elif fc >= 1: add("Scripted Casual Voice", f"{fc} scripted casual phrase(s) — AI mimicking informal voice",                      "ai", "medium", +0.50)

    # A3. Elevated vocabulary density (in non-academic writing context)
    ed = feat["elevated_density"]
    if   ed > 2.5: add("Elevated Vocabulary", f"{feat['elevated_hits']} sophisticated words in context — AI sounds polished even when casual", "ai", "high",   +0.75)
    elif ed > 1.0: add("Elevated Vocabulary", f"{feat['elevated_hits']} elevated words — moderately suspicious",                               "ai", "medium", +0.35)

    # A4. Epistemic hedge density
    hd = feat["hedge_density"]
    if   hd > 2.0: add("Epistemic Hedges", f"{feat['hedge_hits']} analytical hedge phrases ('it is crucial', 'research suggests'…)", "ai", "high",   +0.70)
    elif hd > 0.8: add("Epistemic Hedges", f"{feat['hedge_hits']} hedge phrase(s) — moderate AI indicator",                          "ai", "medium", +0.30)

    # A5. Vague quantifiers (AI says "many" and "various"; humans are specific)
    vq = feat["vague_quantifiers"]
    sn = feat["specific_numbers"]
    if vq >= 3 and sn == 0:
        add("Vague + No Specifics", f"{vq} vague quantifiers ('many', 'various', 'numerous') with zero specific numbers — AI generalises", "ai", "high", +0.65)
    elif vq >= 2 and sn == 0:
        add("Vague Quantifiers",    f"{vq} vague quantifiers, no specific figures",                                                         "ai", "medium", +0.35)

    # A6. Tense uniformity — ONLY fire for analytical/argumentative writing
    # Factual/descriptive writing (human OR AI) legitimately stays present tense
    # So only penalise if ALSO no contractions, no slang, no fragments (= not casual human)
    tm = feat["tense_mix"]
    is_formal_context = feat["contraction_rate"] < 0.005 and feat["slang_hits"] == 0
    if tm < 0.05 and feat["n_sentences"] > 6 and is_formal_context:
        add("Tense Uniformity", f"Single tense throughout in formal context (mix={tm:.2f}) — AI pattern", "ai", "low", +0.25)
    elif tm > 0.30:
        add("Tense Variation", f"Natural tense mixing (score {tm:.2f})",  "human", "low", -0.15)

    # A7. "We" pronoun in non-collaborative writing (AI instructional)
    if feat["we_ratio"] > 0.20 and feat["i_count"] == 0:
        add("Collective 'We' Pronoun", f"Frequent 'we' ({feat['we_count']} times) with no 'I' — AI instructional/analytical voice", "ai", "medium", +0.40)

    # NEW A8. Encyclopedic / Wikipedia-style AI patterns
    eh = feat["encyclopedic_hits"]
    if   eh >= 5: add("Encyclopedic Phrasing", f"{eh} Wikipedia-style phrases ('primarily found in', 'play an important role', 'throughout history'…)", "ai", "high",   +0.90)
    elif eh >= 3: add("Encyclopedic Phrasing", f"{eh} encyclopedic AI patterns detected",                                                               "ai", "high",   +0.70)
    elif eh >= 1: add("Encyclopedic Phrasing", f"{eh} encyclopedic phrase(s) — mild AI indicator",                                                      "ai", "medium", +0.30)

    # NEW A9. Vocabulary sophistication (avg syllables per word)
    avs = feat["avg_syllables"]
    if   avs > 1.6: add("Vocabulary Sophistication", f"Avg {avs:.2f} syllables/word — elevated vocabulary typical of AI factual writing", "ai", "high",   +0.70)
    elif avs > 1.45: add("Vocabulary Sophistication", f"Avg {avs:.2f} syllables/word — moderately formal vocabulary",                     "ai", "medium", +0.35)
    elif avs < 1.25: add("Vocabulary Sophistication", f"Avg {avs:.2f} syllables/word — simple, direct vocabulary, human-like",            "human", "medium", -0.40)

    # NEW A10. Clause complexity — commas per sentence
    cps = feat["comma_per_sent"]
    if   cps > 1.5: add("Clause Complexity", f"{cps:.1f} commas/sentence — AI writes dense subordinate clauses", "ai", "high",   +0.75)
    elif cps > 0.8: add("Clause Complexity", f"{cps:.1f} commas/sentence — moderately complex structure",        "ai", "medium", +0.30)
    elif cps < 0.2 and feat["n_sentences"] > 4:
        add("Low Clause Complexity", f"Very few commas ({cps:.1f}/sent) — simple sentence structure, human/student writing", "human", "medium", -0.45)

    # NEW A11. Adjective-noun pair density (AI packs descriptive adjectives)
    and_ = feat["adj_noun_density"]
    if   and_ > 1.2: add("Descriptive Density", f"High adj-noun density ({and_:.1f}) — AI packs in descriptive adjectives ('impressive manes', 'magnificent animals')", "ai", "high",   +0.70)
    elif and_ > 0.7: add("Descriptive Density", f"Moderate adj-noun density ({and_:.1f})",                                                                               "ai", "medium", +0.25)
    elif and_ < 0.3: add("Descriptive Density", f"Low adj-noun density ({and_:.1f}) — plain writing, human indicator",                                                   "human", "low",  -0.25)

    # NEW A12. Repeated subject starters (student/human writing pattern)
    rsr = feat["repeated_starter_ratio"]
    if rsr > 0.4:
        add("Repeated Subject Opener", f"Same word starts {rsr*100:.0f}% of sentences — student/human writing pattern (not AI)", "human", "medium", -0.55)

    # A8. Robotic sentence openers
    rr = feat["robotic_ratio"]
    if   rr > 0.60: add("Sentence Openers", f"{rr*100:.0f}% start with The/This/It/Furthermore/However…", "ai", "medium", +0.55)
    elif rr > 0.35: add("Sentence Openers", f"{rr*100:.0f}% robotic openers — moderate AI pattern",       "ai", "low",    +0.20)
    elif rr < 0.15: add("Sentence Openers", f"Diverse openers — only {rr*100:.0f}% robotic",              "human", "low", -0.20)

    # A9. Passive voice
    pv = feat["passive_ratio"]
    if   pv > 0.40: add("Passive Voice", f"High ({pv:.2f}/sentence) — AI over-relies on passive constructions", "ai", "medium", +0.40)
    elif pv > 0.20: add("Passive Voice", f"Moderate ({pv:.2f}/sentence)",                                       "mixed", "low",    +0.10)
    else:           add("Passive Voice", f"Low ({pv:.2f}/sentence) — active voice",                             "human", "low",    -0.15)

    # A10. Perplexity — supporting evidence only (lower weight)
    ppl = feat["overall_ppl"]
    pstd = feat["ppl_std"]
    if ppl < 30 and pstd < 10:
        add("Perplexity", f"Very low & uniform (PPL={ppl:.1f}, σ={pstd:.1f}) — strong corroborating AI signal", "ai", "medium", +0.55)
    elif ppl < 50 and pstd < 20:
        add("Perplexity", f"Low perplexity (PPL={ppl:.1f}, σ={pstd:.1f}) — moderate AI corroboration",          "ai", "low",    +0.25)
    elif ppl > 150:
        add("Perplexity", f"High perplexity (PPL={ppl:.1f}) — varied word choices",                             "human", "low",  -0.20)

    # ════════════════════════════════════════════════════════
    # GROUP B: HIGH-PRECISION HUMAN SIGNALS
    # These almost never appear in AI-generated text
    # ════════════════════════════════════════════════════════

    # B1. Internet slang — AI never produces these
    sh = feat["slang_hits"]
    if   sh >= 2: add("Internet Slang", f"{sh} slang terms (lol, btw, yeah, gonna…) — AI never produces these", "human", "high",   -0.90)
    elif sh == 1: add("Internet Slang", "1 slang term — human indicator",                                        "human", "medium", -0.40)

    # B2. Lowercase sentence starts — AI always capitalises
    lr = feat["lowercase_ratio"]
    if   lr > 0.50: add("Lowercase Starts", f"{lr*100:.0f}% sentences start lowercase — AI always capitalises", "human", "high",   -0.90)
    elif lr > 0.15: add("Lowercase Starts", f"Some lowercase starts ({lr*100:.0f}%) — casual human indicator",  "human", "medium", -0.40)

    # B3. Missing-apostrophe contractions (casual human typing)
    ma = feat["missing_apos"]
    if   ma >= 2: add("Informal Contractions", f"{ma} words like 'its/dont/cant' — casual typing, AI never does this", "human", "high",   -0.75)
    elif ma == 1: add("Informal Contractions", "1 apostrophe-less contraction",                                         "human", "medium", -0.35)

    # B4. Sentence fragments
    fr = feat["fragment_ratio"]
    if   fr > 0.20: add("Sentence Fragments", f"{fr*100:.0f}% fragments (<5 words) — natural human writing style", "human", "medium", -0.50)
    elif fr > 0.08: add("Sentence Fragments", f"Some fragments ({fr*100:.0f}%)",                                    "human", "low",    -0.20)
    elif fr == 0 and feat["n_sentences"] >= 6:
        add("No Fragments", "Zero sentence fragments — AI writes in complete sentences", "ai", "low", +0.20)

    # B5. Sentence length variation
    cv = feat["cv_sl"]
    if   cv < 0.15 and feat["n_sentences"] > 5:
        add("Sentence Length Variance", f"Extremely uniform (CV={cv:.2f}) — AI metronomic pattern",          "ai",    "medium", +0.50)
    elif cv < 0.28:
        add("Sentence Length Variance", f"Low variance (CV={cv:.2f})",                                       "ai",    "low",    +0.15)
    elif cv > 0.55:
        add("Sentence Length Variance", f"High variance (CV={cv:.2f}) — humans mix long and short naturally","human", "medium", -0.35)

    # B6. Specificity (humans use numbers, AI uses vague language)
    if feat["specific_numbers"] >= 2:
        add("Specific Details", f"{feat['specific_numbers']} specific numbers/figures — humans are concrete", "human", "low", -0.25)

    # B7. Questions  
    if feat["q_rate"] > 0.15:
        add("Question Usage", f"Questions in {feat['q_rate']*100:.0f}% of sentences — conversational human style", "human", "low", -0.20)

    # B8. Punctuation expressiveness
    if feat["exclaim_rate"] > 0.003:
        add("Exclamation Marks", "Present — emotional expressiveness, human indicator",  "human", "low", -0.20)
    if feat["ellipsis_rate"] > 0.05:
        add("Ellipses",          "Present — hesitation/trailing, human writing style",  "human", "low", -0.15)

    # B9. Genuine contractions
    cr = feat["contraction_rate"]
    if   cr > 0.025: add("Contractions", f"Rate {cr*100:.1f}% — natural informal voice",      "human", "medium", -0.40)
    elif cr > 0.010: add("Contractions", f"Rate {cr*100:.1f}% — moderate informality",        "human", "low",    -0.15)
    elif cr == 0 and feat["n_words"] > 60:
        add("No Contractions", "Zero contractions in long text — formal/AI register",          "ai",    "medium", +0.35)

    # B10. Lexical diversity
    mt = feat["msttr"]
    if   mt > 0.85: add("Lexical Diversity", f"High MSTTR {mt:.3f} — rich diverse vocabulary", "human", "low", -0.20)
    elif mt < 0.60: add("Lexical Diversity", f"Low MSTTR {mt:.3f} — repetitive vocabulary",    "ai",    "low", +0.20)

    # ════════════════════════════════════════════════════════
    # GROUP C: CONTEXTUAL SIGNALS (lower weight, used for tiebreak)
    # ════════════════════════════════════════════════════════

    # C1. Paragraph regularity
    if feat["cv_para"] < 0.08 and feat["n_sentences"] > 8:
        add("Paragraph Regularity", "Paragraphs nearly identical in length — structural AI pattern", "ai", "low", +0.20)

    # C2. N-gram repetition
    rs = feat["repeat_score"]
    if rs > 0.18: add("Phrase Repetition", f"High bigram repetition ({rs:.2f})", "ai", "low", +0.20)

    # C3. Flesch readability
    fl = feat["flesch"]
    if   fl > 80:       add("Readability", f"Very easy Flesch {fl:.0f} — simplified/padded", "ai", "low", +0.15)
    elif 48 <= fl <= 68: add("Readability", f"Standard Flesch {fl:.0f} — AI clusters here",  "ai", "low", +0.10)
    elif fl < 25:        add("Readability", f"Very difficult Flesch {fl:.0f} — dense text",  "human", "low", -0.10)

    # ── Normalise ──────────────────────────────────────────────────────────
    if total_w == 0:
        return 0.5, sigs
    raw     = score / total_w
    ai_prob = 1 / (1 + math.exp(-raw * 3.2))
    return round(ai_prob, 4), sigs


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE FLAGGING
# ─────────────────────────────────────────────────────────────────────────────
def flag_sentences(sentences, sent_ppls) -> list[dict]:
    flagged   = []
    threshold = float(np.percentile(sent_ppls, 25)) + 5 if len(sent_ppls) >= 4 else 45.0

    for sent, ppl in zip(sentences, sent_ppls):
        reasons = []
        if any(p.search(sent.lower()) for p in _FORMAL_RE):
            reasons.append("formal AI transition phrase")
        if any(p.search(sent.lower()) for p in _CASUAL_RE):
            reasons.append("scripted casual phrase")
        if _ELEVATED.search(sent):
            hits = _ELEVATED.findall(sent)
            reasons.append(f"elevated vocabulary: '{hits[0]}'")
        if _HEDGES.search(sent):
            reasons.append("epistemic hedge phrase")
        if ppl < threshold and ppl < 60 and not reasons:
            reasons.append(f"unusually predictable phrasing (perplexity {ppl:.1f})")
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

    bi_lm  = NgramLM(2); bi_lm.train(text)
    tri_lm = NgramLM(3); tri_lm.train(text)

    feat             = extract_features(text, bi_lm, tri_lm)
    ai_prob, signals = score_features(feat)

    # Hardened thresholds: require stronger evidence
    if   ai_prob > 0.72: classification = "AI Generated"
    elif ai_prob > 0.40: classification = "Mixed / Uncertain"
    else:                classification = "Human Written"

    confidence = int(abs(ai_prob - 0.5) * 2 * 100)

    sentences = sent_tokenize(text)
    sent_ppls = feat["sent_ppls"]
    sent_data = [{"text": s, "perplexity": round(p, 2)}
                 for s, p in zip(sentences, sent_ppls)]

    ppl = feat["overall_ppl"]
    human_like = round(min(max((ppl - 10) / 590, 0), 1) * 100, 1)

    if   ppl < 35:  ppl_interp = "Very predictable — possible AI indicator"
    elif ppl < 80:  ppl_interp = "Moderately predictable — mixed signals"
    elif ppl < 200: ppl_interp = "Variable — leans human"
    else:           ppl_interp = "Highly varied — human-like"

    flagged    = flag_sentences(sentences, sent_ppls)
    ai_sigs    = [s for s in signals if s["direction"] == "ai"    and s["weight"] in ("high","medium")]
    human_sigs = [s for s in signals if s["direction"] == "human" and s["weight"] in ("high","medium")]

    if   classification == "AI Generated":
        top     = ", ".join(s["signal"].lower() for s in ai_sigs[:2])
        summary = f"This text was likely AI-generated ({int(ai_prob*100)}% probability). Key indicators: {top}." if top else f"AI-generated ({int(ai_prob*100)}%)."
    elif classification == "Human Written":
        top     = ", ".join(s["signal"].lower() for s in human_sigs[:2])
        summary = f"This text reads as human-written ({int((1-ai_prob)*100)}% confidence). Key signals: {top}." if top else f"Human-written ({int((1-ai_prob)*100)}%)."
    else:
        summary = "Mixed signals — could be AI-assisted, lightly edited AI output, or structured human writing."

    radar = {
        "AI Phrases":    round(min(feat["all_ai_phrase_hits"] / 5, 1), 2),
        "Elevated Vocab":round(min(feat["elevated_density"]   / 3, 1), 2),
        "Formality":     round(max(0, 1 - feat["contraction_rate"] / 0.03), 2),
        "Uniformity":    round(max(0, 1 - feat["cv_sl"] / 0.6), 2),
        "Hedging":       round(min(feat["hedge_density"] / 2, 1), 2),
        "Specificity":   round(min(feat["specificity"] / 2, 1), 2),
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
            "fog_index": round(feat["fog"],    1),
            "msttr":     round(feat["msttr"],  3),
        },
        "signals":           signals,
        "flagged_sentences": flagged,
        "radar":             radar,
        "stats": {
            "word_count":          feat["n_words"],
            "sentence_count":      feat["n_sentences"],
            "char_count":          len(text),
            "ttr":                 round(feat["msttr"], 3),
            "formal_phrases":      feat["formal_hits"],
            "fake_casual":         feat["fake_casual_hits"],
            "elevated_vocab":      feat["elevated_hits"],
            "hedge_phrases":       feat["hedge_hits"],
            "slang_hits":          feat["slang_hits"],
            "informal_contractions": feat["missing_apos"],
            "avg_sentence_len":    round(feat["avg_sl"], 1),
            "fragment_ratio":      round(feat["fragment_ratio"], 2),
            "tense_mix":           round(feat["tense_mix"], 2),
            "specificity":         feat["specific_numbers"],
            "vague_quantifiers":   feat["vague_quantifiers"],
            "encyclopedic_hits":   feat["encyclopedic_hits"],
            "avg_syllables":       round(feat["avg_syllables"], 2),
            "comma_per_sent":      round(feat["comma_per_sent"], 1),
        },
    }