"""
app.py  —  TextGuard v2: Upgraded AI Detection Dashboard
100% local · no API keys · trigram LM + 18 linguistic features
"""

import streamlit as st
import html as _html
import sys, os, math, json, base64
from datetime import datetime

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TextGuard — AI Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:      #0B0F1A; --surf:   #131929; --surf2:  #1C2438;
    --border:  #2A3550; --indigo: #6366F1; --indigo2:#818CF8;
    --amber:   #F59E0B; --green:  #10B981; --red:    #EF4444;
    --slate:   #94A3B8; --white:  #E2E8F0; --r:12px;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;font-family:'Inter',sans-serif;color:var(--white);}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stSidebar"]{background:var(--surf)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stMain"]>div{padding-top:0!important;}
#MainMenu,footer,header{visibility:hidden;}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;}

[data-baseweb="tab-list"]{background:var(--surf2)!important;border-radius:var(--r)!important;padding:4px!important;border:1px solid var(--border)!important;gap:4px!important;}
[data-baseweb="tab"]{font-family:'Space Grotesk',sans-serif!important;font-weight:500!important;color:var(--slate)!important;border-radius:8px!important;padding:8px 22px!important;}
[aria-selected="true"][data-baseweb="tab"]{background:var(--indigo)!important;color:#fff!important;}

[data-testid="stButton"]>button{background:var(--indigo)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-family:'Space Grotesk',sans-serif!important;font-weight:600!important;font-size:15px!important;padding:12px 28px!important;transition:all .18s!important;}
[data-testid="stButton"]>button:hover{background:var(--indigo2)!important;transform:translateY(-1px)!important;box-shadow:0 4px 20px rgba(99,102,241,.4)!important;}

[data-testid="stTextArea"] textarea{background:var(--surf2)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important;color:var(--white)!important;font-family:'Inter',sans-serif!important;font-size:14px!important;}
[data-testid="stTextArea"] textarea:focus{border-color:var(--indigo)!important;box-shadow:0 0 0 3px rgba(99,102,241,.2)!important;}
[data-testid="stFileUploader"]{background:var(--surf2)!important;border:2px dashed var(--border)!important;border-radius:var(--r)!important;}
[data-testid="stMetric"]{background:var(--surf2)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important;padding:16px!important;}
[data-testid="stMetricLabel"]{color:var(--slate)!important;font-family:'Space Grotesk',sans-serif!important;}
[data-testid="stMetricValue"]{color:var(--white)!important;font-family:'Space Grotesk',sans-serif!important;}
[data-testid="stExpander"]{background:var(--surf2)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important;}
[data-testid="stSelectbox"] > div > div{background:var(--surf2)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--white)!important;}

::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:99px;}
code{font-family:'JetBrains Mono',monospace!important;font-size:13px!important;}

.stAlert{border-radius:var(--r)!important;}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 4px 12px;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:#E2E8F0;">
            🛡️ TextGuard
        </div>
        <div style="font-size:11px;color:#94A3B8;margin-top:2px;">AI Detection & Analysis v2</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    mode = st.radio("Mode", ["🔍 Single Analysis", "⚖️ Compare Two Texts"], label_visibility="collapsed")
    st.markdown("---")

    st.markdown("<p style='font-size:12px;color:#94A3B8;font-family:Space Grotesk,sans-serif;font-weight:600;letter-spacing:.06em;text-transform:uppercase;'>Detection Engine</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px;color:#64748B;line-height:1.8;">
    ✦ Trigram + Bigram LM<br>
    ✦ Perplexity burstiness<br>
    ✦ AI hallmark phrases (50+)<br>
    ✦ Shannon entropy<br>
    ✦ Flesch + Gunning Fog<br>
    ✦ Passive voice detection<br>
    ✦ Sentence opener diversity<br>
    ✦ Named entity density<br>
    ✦ N-gram repetition<br>
    ✦ Punctuation fingerprint<br>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='font-size:11px;color:#475569;text-align:center;'>100% local · no API keys · no internet</p>", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_analyzer():
    from utils.analyzer import analyze_text
    return analyze_text

@st.cache_resource(show_spinner=False)
def load_file_processor():
    from utils.file_handler import process_file
    return process_file

# Cache analysis results by text hash — avoids re-running on every Streamlit rerun
@st.cache_data(show_spinner=False, max_entries=20)
def cached_analyze(text: str) -> dict:
    fn = load_analyzer()
    return fn(text)

CLS_STYLE = {
    "AI Generated":      ("#EF4444", "#200A0A"),
    "Human Written":     ("#10B981", "#0A2218"),
    "Mixed / Uncertain": ("#F59E0B", "#241A06"),
}

def render_verdict_banner(R):
    cls   = R["classification"]
    prob  = R["ai_probability"]
    conf  = R["confidence"]
    summ  = R["summary"]
    cc, cbg = CLS_STYLE.get(cls, ("#94A3B8","#1C2438"))
    icon = {"AI Generated":"🤖","Human Written":"🧑","Mixed / Uncertain":"🔀"}.get(cls,"")
    st.markdown(f"""
    <div style="background:{cbg};border:1px solid {cc};border-radius:16px;
        padding:24px 28px;display:flex;align-items:center;
        justify-content:space-between;flex-wrap:wrap;gap:14px;margin-bottom:24px;">
        <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:11px;color:{cc};
                letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px;">Verdict</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:700;color:{cc};">
                {icon} {cls}</div>
            <div style="font-family:'Inter',sans-serif;font-size:13px;color:#94A3B8;
                margin-top:7px;max-width:560px;line-height:1.5;">
                {_html.escape(summ)}</div>
        </div>
        <div style="text-align:center;">
            <div style="width:84px;height:84px;border-radius:50%;border:3px solid {cc};
                display:flex;flex-direction:column;align-items:center;justify-content:center;
                background:{cbg};">
                <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:{cc};">
                    {conf}%</div>
                <div style="font-size:9px;color:#94A3B8;">confidence</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

def render_radar(radar: dict, label: str = ""):
    """SVG spider/radar chart — pure HTML/SVG, no dependencies."""
    keys = list(radar.keys())
    vals = list(radar.values())
    n    = len(keys)
    cx, cy, r_max = 160, 160, 110
    angles = [math.pi/2 - 2*math.pi*i/n for i in range(n)]

    def pt(angle, radius):
        return cx + radius*math.cos(angle), cy - radius*math.sin(angle)

    # Grid rings
    rings = ""
    for frac in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{pt(a, r_max*frac)[0]:.1f},{pt(a, r_max*frac)[1]:.1f}" for a in angles)
        rings += f'<polygon points="{pts}" fill="none" stroke="#2A3550" stroke-width="1"/>'

    # Spokes
    spokes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{pt(a,r_max)[0]:.1f}" y2="{pt(a,r_max)[1]:.1f}" stroke="#2A3550" stroke-width="1"/>'
        for a in angles)

    # Data polygon
    data_pts = " ".join(f"{pt(a, r_max*v)[0]:.1f},{pt(a, r_max*v)[1]:.1f}"
                        for a, v in zip(angles, vals))
    data_poly = f'<polygon points="{data_pts}" fill="rgba(99,102,241,0.25)" stroke="#6366F1" stroke-width="2"/>'
    data_dots = "".join(
        f'<circle cx="{pt(a,r_max*v)[0]:.1f}" cy="{pt(a,r_max*v)[1]:.1f}" r="4" fill="#6366F1"/>'
        for a, v in zip(angles, vals))

    # Labels
    labels_svg = ""
    for i, (key, angle) in enumerate(zip(keys, angles)):
        lx, ly = pt(angle, r_max + 22)
        anchor = "middle" if abs(math.cos(angle)) < 0.3 else ("start" if math.cos(angle) > 0 else "end")
        labels_svg += (f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
                       f'font-family="Space Grotesk,sans-serif" font-size="10" fill="#94A3B8">{key}</text>')

    title_svg = f'<text x="{cx}" y="18" text-anchor="middle" font-family="Space Grotesk,sans-serif" font-size="11" fill="#64748B">{label}</text>' if label else ""

    svg = f"""<svg viewBox="0 0 320 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:320px;">
        {rings}{spokes}{data_poly}{data_dots}{labels_svg}{title_svg}
    </svg>"""
    return svg

def render_perplexity_bars(ppl_data: dict):
    sents = ppl_data.get("sentences", [])
    if not sents: return
    max_p = max(s["perplexity"] for s in sents) or 1
    bars  = ""
    for s in sents[:30]:
        pct = min(s["perplexity"]/max_p*100, 100)
        col = "#EF4444" if s["perplexity"] < 40 else ("#F59E0B" if s["perplexity"] < 100 else "#10B981")
        prev = _html.escape(s["text"][:70]) + ("…" if len(s["text"])>70 else "")
        bars += f"""
        <div style="margin-bottom:7px;" title="{_html.escape(s['text'])}">
          <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
            <span style="font-size:11px;color:#94A3B8;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;max-width:82%;">{prev}</span>
            <span style="font-size:11px;color:{col};font-family:'JetBrains Mono',monospace;white-space:nowrap;margin-left:6px;">{s['perplexity']:.1f}</span>
          </div>
          <div style="background:#2A3550;border-radius:99px;height:5px;">
            <div style="background:{col};width:{pct:.1f}%;height:5px;border-radius:99px;"></div>
          </div>
        </div>"""
    st.markdown(f"""
    <div style="background:#131929;border:1px solid #2A3550;border-radius:12px;padding:16px 18px;max-height:380px;overflow-y:auto;">
      <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
        <span style="font-size:12px;color:#94A3B8;">
          Bigram: <strong style="color:#E2E8F0;">{ppl_data['bigram']:.1f}</strong> &nbsp;·&nbsp;
          Trigram: <strong style="color:#E2E8F0;">{ppl_data['trigram']:.1f}</strong> &nbsp;·&nbsp;
          σ: <strong style="color:#E2E8F0;">{ppl_data['ppl_std']:.1f}</strong>
        </span>
        <span style="font-size:11px;color:#94A3B8;font-style:italic;">{ppl_data['interpretation']}</span>
      </div>
      {bars}
    </div>
    <div style="display:flex;gap:14px;margin-top:8px;">
      <span style="font-size:11px;color:#EF4444;">🔴 Low — AI-like</span>
      <span style="font-size:11px;color:#F59E0B;">🟡 Medium</span>
      <span style="font-size:11px;color:#10B981;">🟢 High — Human</span>
    </div>""", unsafe_allow_html=True)

def render_signals_panel(signals: list):
    wc  = {"high":"#EF4444","medium":"#F59E0B","low":"#10B981"}
    dic = {"ai":"🤖","human":"🧑","mixed":"🔀"}
    cards = ""
    for s in signals:
        c = wc.get(s["weight"],"#94A3B8")
        cards += f"""
        <div style="background:#131929;border:1px solid #2A3550;border-radius:9px;padding:11px 13px;margin-bottom:8px;">
          <div style="display:flex;align-items:center;gap:7px;margin-bottom:4px;">
            <span style="width:7px;height:7px;border-radius:50%;background:{c};flex-shrink:0;"></span>
            <span style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:600;color:#E2E8F0;">{_html.escape(s['signal'])}</span>
            <span style="margin-left:auto;font-size:10px;color:{c};text-transform:uppercase;letter-spacing:.06em;">{s['weight']}</span>
            <span style="font-size:12px;">{dic.get(s['direction'],'')}</span>
          </div>
          <div style="font-family:'Inter',sans-serif;font-size:12px;color:#94A3B8;line-height:1.45;">
            {_html.escape(s['observation'])}</div>
        </div>"""
    st.markdown(f'<div style="max-height:400px;overflow-y:auto;">{cards}</div>', unsafe_allow_html=True)

def render_flagged(flags: list):
    if not flags:
        st.success("No sentences flagged.")
        return
    for fs in flags:
        reasons = " · ".join(_html.escape(r) for r in fs["reasons"])
        st.markdown(f"""
        <div style="background:#1A0808;border-left:3px solid #EF4444;border-radius:0 8px 8px 0;
            padding:12px 15px;margin-bottom:9px;">
          <div style="font-family:'Inter',sans-serif;font-size:13px;color:#E2E8F0;
              margin-bottom:5px;line-height:1.5;">
            &ldquo;{_html.escape(fs['text'])}&rdquo;</div>
          <div style="font-size:11px;color:#EF4444;">⚠️ {reasons} &nbsp;·&nbsp; ppl: {fs['perplexity']}</div>
        </div>""", unsafe_allow_html=True)

def render_readability(R):
    rd = R["readability"]
    fl = rd["flesch"]
    if   fl > 80: fl_label, fl_c = "Very Easy", "#F59E0B"
    elif fl > 60: fl_label, fl_c = "Easy",      "#10B981"
    elif fl > 40: fl_label, fl_c = "Standard",  "#94A3B8"
    else:         fl_label, fl_c = "Difficult",  "#6366F1"

    fog = rd["fog_index"]
    if   fog > 17: fog_c = "#EF4444"
    elif fog > 12: fog_c = "#F59E0B"
    else:          fog_c = "#10B981"

    st.markdown(f"""
    <div style="background:#131929;border:1px solid #2A3550;border-radius:12px;padding:18px 20px;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;
          color:#E2E8F0;margin-bottom:14px;">📖 Readability Scores</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">
        <div style="text-align:center;">
          <div style="font-size:26px;font-weight:700;color:{fl_c};font-family:'Space Grotesk',sans-serif;">{fl:.0f}</div>
          <div style="font-size:11px;color:#94A3B8;">Flesch Ease</div>
          <div style="font-size:10px;color:{fl_c};">{fl_label}</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:26px;font-weight:700;color:{fog_c};font-family:'Space Grotesk',sans-serif;">{fog:.1f}</div>
          <div style="font-size:11px;color:#94A3B8;">Gunning Fog</div>
          <div style="font-size:10px;color:{fog_c};">Grade {fog:.0f}</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:26px;font-weight:700;color:#818CF8;font-family:'Space Grotesk',sans-serif;">{rd['msttr']:.3f}</div>
          <div style="font-size:11px;color:#94A3B8;">MSTTR</div>
          <div style="font-size:10px;color:#818CF8;">Lexical diversity</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

def render_authorship_bar(prob):
    ai_pct = round(prob*100, 1)
    hu_pct = round(100-ai_pct, 1)
    st.markdown(f"""
    <div style="background:#131929;border:1px solid #2A3550;border-radius:12px;
        padding:16px 20px;margin-bottom:20px;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:600;
          color:#E2E8F0;margin-bottom:10px;">Authorship Distribution</div>
      <div style="display:flex;height:12px;border-radius:99px;overflow:hidden;margin-bottom:6px;">
        <div style="background:#EF4444;width:{ai_pct}%;"></div>
        <div style="background:#10B981;width:{hu_pct}%;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;">
        <span style="font-size:12px;color:#EF4444;">🤖 AI: {ai_pct}%</span>
        <span style="font-size:12px;color:#10B981;">🧑 Human: {hu_pct}%</span>
      </div>
    </div>""", unsafe_allow_html=True)

def make_report(R, label=""):
    s = R["stats"]; p = R["perplexity"]; rd = R["readability"]
    lines = [
        f"TextGuard Analysis Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"{'='*50}",
        f"Source: {label}" if label else "",
        f"\nVERDICT: {R['classification']}",
        f"AI Probability: {round(R['ai_probability']*100,1)}%",
        f"Confidence: {R['confidence']}%",
        f"\nSUMMARY\n{R['summary']}",
        f"\nPERPLEXITY\nOverall: {p['overall']} (Bigram: {p['bigram']}, Trigram: {p['trigram']})",
        f"Interpretation: {p['interpretation']}",
        f"\nREADABILITY\nFlesch: {rd['flesch']}  |  Gunning Fog: {rd['fog_index']}  |  MSTTR: {rd['msttr']}",
        f"\nSTATS\nWords: {s['word_count']}  |  Sentences: {s['sentence_count']}",
        f"TTR: {s['ttr']}  |  AI Phrases: {s.get('formal_phrases',0) + s.get('fake_casual',0)}",
        f"Avg Sentence Length: {s['avg_sentence_len']} words",
        f"\nFLAGGED SENTENCES ({len(R['flagged_sentences'])})",
    ]
    for i, fs in enumerate(R["flagged_sentences"], 1):
        lines.append(f'{i}. "{fs["text"]}"\n   → {"; ".join(fs["reasons"])}')
    lines += [f"\nDETECTION SIGNALS ({len(R['signals'])})"]
    for sig in R["signals"]:
        lines.append(f"[{sig['weight'].upper()}] {sig['signal']}: {sig['observation']}")
    return "\n".join(l for l in lines if l is not None)

def full_results(R, label=""):
    render_verdict_banner(R)
    render_authorship_bar(R["ai_probability"])

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    with m1: st.metric("AI Probability",   f"{round(R['ai_probability']*100,1)}%")
    with m2: st.metric("Perplexity",        f"{R['perplexity']['overall']:.1f}")
    with m3: st.metric("Human-likeness",    f"{R['perplexity']['normalized']:.0f}/100")
    with m4: st.metric("Words",             R['stats']['word_count'])
    with m5: st.metric("AI Phrases",        R['stats'].get('formal_phrases', 0) + R['stats'].get('fake_casual', 0))
    with m6: st.metric("Flagged Sentences", len(R['flagged_sentences']))

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # Radar + readability side by side
    rc1, rc2 = st.columns([1, 1], gap="large")
    with rc1:
        st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:16px;
            font-weight:600;color:#E2E8F0;margin-bottom:10px;">🕸️ Signal Radar</div>""",
            unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:#131929;border:1px solid #2A3550;border-radius:12px;'
            f'padding:12px;display:flex;justify-content:center;">'
            f'{render_radar(R["radar"], label)}</div>',
            unsafe_allow_html=True)
    with rc2:
        render_readability(R)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        # Extra stats card
        s = R["stats"]
        st.markdown(f"""
        <div style="background:#131929;border:1px solid #2A3550;border-radius:12px;padding:16px 18px;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:600;
              color:#E2E8F0;margin-bottom:12px;">📋 Text Stats</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;color:#94A3B8;">
            <div>Avg Sentence Length<br><strong style="color:#E2E8F0;">{s['avg_sentence_len']} words</strong></div>
            <div>Fragment Ratio<br><strong style="color:#E2E8F0;">{round(s['fragment_ratio']*100)}%</strong></div>
            <div>Lexical Diversity (TTR)<br><strong style="color:#E2E8F0;">{s['ttr']:.3f}</strong></div>
            <div>Tense Mix Score<br><strong style="color:#E2E8F0;">{s['tense_mix']:.2f}</strong></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # Perplexity + Signals
    pc1, pc2 = st.columns([3, 2], gap="large")
    with pc1:
        st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:16px;
            font-weight:600;color:#E2E8F0;margin-bottom:10px;">📊 Sentence Perplexity</div>""",
            unsafe_allow_html=True)
        render_perplexity_bars(R["perplexity"])
    with pc2:
        st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:16px;
            font-weight:600;color:#E2E8F0;margin-bottom:10px;">🔬 Signals</div>""",
            unsafe_allow_html=True)
        render_signals_panel(R["signals"])

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # Flagged sentences
    st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:16px;
        font-weight:600;color:#E2E8F0;margin-bottom:10px;">🚩 Flagged Sentences</div>""",
        unsafe_allow_html=True)
    render_flagged(R["flagged_sentences"])

    # Highlighted doc
    with st.expander("📄 Full Text with Highlights"):
        safe = _html.escape(R.get("_input_text","")).replace("\n","<br>")
        for fs in R["flagged_sentences"]:
            orig = _html.escape(fs["text"])
            if orig in safe:
                safe = safe.replace(orig,
                    f'<mark style="background:#3A1010;color:#EF9090;border-radius:3px;padding:0 2px;">{orig}</mark>',1)
        st.markdown(f"""
        <div style="background:#131929;border:1px solid #2A3550;border-radius:12px;
            padding:18px 20px;font-size:13px;color:#CBD5E1;line-height:1.8;
            max-height:340px;overflow-y:auto;">{safe}</div>""", unsafe_allow_html=True)

    # Export
    report_txt = make_report(R, label)
    b64 = base64.b64encode(report_txt.encode()).decode()
    st.markdown(
        f'<a href="data:file/txt;base64,{b64}" download="textguard_report.txt" '
        f'style="display:inline-block;background:#1C2438;border:1px solid #2A3550;'
        f'color:#818CF8;padding:8px 18px;border-radius:8px;font-size:13px;'
        f'text-decoration:none;font-family:Space Grotesk,sans-serif;margin-top:8px;">'
        f'📥 Download Report</a>', unsafe_allow_html=True)

    with st.expander("🔧 Raw JSON"):
        display_R = {k: v for k, v in R.items() if k != "_input_text"}
        st.json(display_R)


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:36px 0 24px;text-align:center;border-bottom:1px solid #2A3550;margin-bottom:32px;">
  <h1 style="font-family:'Space Grotesk',sans-serif;font-size:clamp(26px,4vw,46px);
      font-weight:700;color:#E2E8F0;margin:0 0 8px;letter-spacing:-.02em;line-height:1.1;">
    Detect AI-Generated Text<br><span style="color:#6366F1;">with Forensic Precision</span>
  </h1>
  <p style="font-family:'Inter',sans-serif;color:#94A3B8;font-size:15px;
      max-width:500px;margin:0 auto;line-height:1.6;">
    Trigram LM · 18 linguistic signals · Radar chart · Readability scores<br>
    100% local — no API keys, no internet required.
  </p>
</div>
""", unsafe_allow_html=True)

process_fn = load_file_processor()

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE ANALYSIS MODE
# ─────────────────────────────────────────────────────────────────────────────
if "Single" in mode:
    tab_text, tab_file = st.tabs(["✏️  Paste Text", "📎  Upload File"])
    input_text = ""
    file_label = ""

    with tab_text:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        raw = st.text_area("text", height=220,
            placeholder="Paste any text — essay, email, article, code, assignment…",
            label_visibility="collapsed")
        wc = len(raw.split()) if raw.strip() else 0
        st.markdown(f"<p style='font-size:12px;color:#94A3B8;margin-top:2px;'>{wc} words</p>",
                    unsafe_allow_html=True)
        c1, _ = st.columns([1,4])
        with c1:
            go = st.button("Analyze →", key="go_text", use_container_width=True)
        if go:
            if raw.strip(): input_text, file_label = raw.strip(), "Pasted Text"
            else: st.warning("Paste some text first.")

    with tab_file:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        up = st.file_uploader("upload", label_visibility="collapsed",
            type=["pdf","docx","txt","py","js","ts","jsx","tsx","java","c","cpp","cs",
                  "go","rs","rb","php","swift","kt","html","css","json","sql","md","sh","yaml","yml"])
        c2, _ = st.columns([1,4])
        with c2:
            go2 = st.button("Analyze →", key="go_file", use_container_width=True)
        if go2:
            if up:
                try:
                    txt, ftype = process_fn(up.name, up.read())
                    if txt.strip(): input_text, file_label = txt.strip(), f"{ftype}: {up.name}"
                    else: st.error("No text extracted.")
                except ValueError as e: st.error(str(e))
            else: st.warning("Upload a file first.")

    if input_text:
        st.markdown("<hr style='border:none;border-top:1px solid #2A3550;margin:28px 0;'>",
                    unsafe_allow_html=True)
        with st.spinner("Running analysis…"):
            R = cached_analyze(input_text)
            R = dict(R); R["_input_text"] = input_text
        if "error" in R: st.error(R["error"])
        else: full_results(R, file_label)

# ─────────────────────────────────────────────────────────────────────────────
# COMPARE MODE
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="background:#131929;border:1px solid #2A3550;border-radius:12px;
        padding:14px 18px;margin-bottom:20px;font-size:13px;color:#94A3B8;">
        ⚖️ <strong style="color:#E2E8F0;">Compare Mode</strong> — paste two texts side by side
        to compare their AI detection scores, perplexity profiles, and linguistic fingerprints.
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    text_a = text_b = ""

    with col_a:
        st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:14px;
            font-weight:600;color:#E2E8F0;margin-bottom:8px;">Text A</div>""",
            unsafe_allow_html=True)
        text_a = st.text_area("Text A", height=200, label_visibility="collapsed",
            key="cmp_a", placeholder="Paste first text here…")

    with col_b:
        st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:14px;
            font-weight:600;color:#E2E8F0;margin-bottom:8px;">Text B</div>""",
            unsafe_allow_html=True)
        text_b = st.text_area("Text B", height=200, label_visibility="collapsed",
            key="cmp_b", placeholder="Paste second text here…")

    cc1, _ = st.columns([1,4])
    with cc1:
        go_cmp = st.button("Compare Both →", key="go_cmp", use_container_width=True)

    if go_cmp:
        if not text_a.strip() or not text_b.strip():
            st.warning("Please fill in both text boxes.")
        else:
            with st.spinner("Analyzing both texts…"):
                Ra = dict(cached_analyze(text_a.strip())); Ra["_input_text"] = text_a.strip()
                Rb = dict(cached_analyze(text_b.strip())); Rb["_input_text"] = text_b.strip()

            st.markdown("<hr style='border:none;border-top:1px solid #2A3550;margin:24px 0;'>",
                        unsafe_allow_html=True)

            # ── Side-by-side summary ──────────────────────────────────────
            s1, s2 = st.columns(2, gap="large")
            for col, R, lbl in [(s1, Ra, "Text A"), (s2, Rb, "Text B")]:
                with col:
                    cls = R["classification"]
                    cc_color, cc_bg = CLS_STYLE.get(cls, ("#94A3B8","#1C2438"))
                    st.markdown(f"""
                    <div style="background:{cc_bg};border:1px solid {cc_color};border-radius:14px;
                        padding:18px 20px;margin-bottom:16px;text-align:center;">
                      <div style="font-size:11px;color:{cc_color};letter-spacing:.1em;
                          text-transform:uppercase;margin-bottom:4px;">{lbl}</div>
                      <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;
                          font-weight:700;color:{cc_color};">{cls}</div>
                      <div style="font-size:28px;font-weight:700;color:{cc_color};margin-top:6px;">
                          {round(R['ai_probability']*100,1)}%
                      </div>
                      <div style="font-size:11px;color:#94A3B8;">AI probability</div>
                    </div>""", unsafe_allow_html=True)

            # ── Radar comparison ──────────────────────────────────────────
            r1, r2 = st.columns(2, gap="large")
            for col, R, lbl in [(r1, Ra, "Text A"), (r2, Rb, "Text B")]:
                with col:
                    st.markdown(f"""<div style="font-family:'Space Grotesk',sans-serif;
                        font-size:14px;font-weight:600;color:#E2E8F0;margin-bottom:8px;">
                        🕸️ {lbl} Radar</div>""", unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="background:#131929;border:1px solid #2A3550;'
                        f'border-radius:12px;padding:10px;display:flex;justify-content:center;">'
                        f'{render_radar(R["radar"])}</div>', unsafe_allow_html=True)

            # ── Metric comparison table ───────────────────────────────────
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:16px;
                font-weight:600;color:#E2E8F0;margin-bottom:12px;">📊 Head-to-Head Metrics</div>""",
                unsafe_allow_html=True)

            def cmp_row(label, va, vb, ai_high=True):
                """Render a comparison row. ai_high=True means higher value = more AI."""
                try:
                    fva, fvb = float(str(va).replace("%","")), float(str(vb).replace("%",""))
                    if ai_high:
                        ca = "#EF4444" if fva > fvb else "#94A3B8"
                        cb = "#EF4444" if fvb > fva else "#94A3B8"
                    else:
                        ca = "#10B981" if fva > fvb else "#94A3B8"
                        cb = "#10B981" if fvb > fva else "#94A3B8"
                except: ca = cb = "#94A3B8"
                return (f'<tr style="border-bottom:1px solid #2A3550;">'
                        f'<td style="padding:10px 14px;font-size:12px;color:#94A3B8;">{label}</td>'
                        f'<td style="padding:10px 14px;font-size:13px;font-weight:600;color:{ca};text-align:right;">{va}</td>'
                        f'<td style="padding:10px 14px;font-size:13px;font-weight:600;color:{cb};text-align:right;">{vb}</td>'
                        f'</tr>')

            rows = (
                cmp_row("AI Probability",       f"{round(Ra['ai_probability']*100,1)}%",  f"{round(Rb['ai_probability']*100,1)}%",  True)
              + cmp_row("Perplexity (lower=AI)", f"{Ra['perplexity']['overall']:.1f}",     f"{Rb['perplexity']['overall']:.1f}",     True)
              + cmp_row("Ppl Std Dev (lower=AI)",f"{Ra['perplexity']['ppl_std']:.1f}",     f"{Rb['perplexity']['ppl_std']:.1f}",     True)
              + cmp_row("Flesch Ease",           f"{Ra['readability']['flesch']:.0f}",     f"{Rb['readability']['flesch']:.0f}",     False)
              + cmp_row("Gunning Fog (higher=AI)",f"{Ra['readability']['fog_index']:.1f}", f"{Rb['readability']['fog_index']:.1f}",  True)
              + cmp_row("MSTTR (lower=AI)",      f"{Ra['readability']['msttr']:.3f}",      f"{Rb['readability']['msttr']:.3f}",      False)
              + cmp_row("AI Phrases",            Ra['stats'].get('formal_phrases',0)+Ra['stats'].get('fake_casual',0), Rb['stats'].get('formal_phrases',0)+Rb['stats'].get('fake_casual',0), True)
              + cmp_row("Flagged Sentences",     len(Ra['flagged_sentences']),             len(Rb['flagged_sentences']),             True)
              + cmp_row("TTR (lower=AI)",        f"{Ra['stats']['ttr']:.3f}",              f"{Rb['stats']['ttr']:.3f}",              False)
            )
            st.markdown(f"""
            <div style="background:#131929;border:1px solid #2A3550;border-radius:12px;overflow:hidden;">
              <table style="width:100%;border-collapse:collapse;">
                <thead>
                  <tr style="background:#1C2438;">
                    <th style="padding:10px 14px;font-family:'Space Grotesk',sans-serif;
                        font-size:11px;color:#64748B;text-align:left;text-transform:uppercase;
                        letter-spacing:.07em;">Metric</th>
                    <th style="padding:10px 14px;font-family:'Space Grotesk',sans-serif;
                        font-size:11px;color:#64748B;text-align:right;text-transform:uppercase;
                        letter-spacing:.07em;">Text A</th>
                    <th style="padding:10px 14px;font-family:'Space Grotesk',sans-serif;
                        font-size:11px;color:#64748B;text-align:right;text-transform:uppercase;
                        letter-spacing:.07em;">Text B</th>
                  </tr>
                </thead>
                <tbody>{rows}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

            # ── Full drill-downs ──────────────────────────────────────────
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            with st.expander("📂 Full Analysis — Text A"):
                full_results(Ra, "Text A")
            with st.expander("📂 Full Analysis — Text B"):
                full_results(Rb, "Text B")