"""
app.py  —  TextGuard: AI Detection & Perplexity Analysis
100% local, no API keys required.
Run: streamlit run app.py
"""

import streamlit as st
import html as html_module
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Plagiarism Tool",
    page_icon="🔍🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:       #0B0F1A;
    --surf:     #131929;
    --surf2:    #1C2438;
    --border:   #2A3550;
    --indigo:   #6366F1;
    --indigo2:  #818CF8;
    --amber:    #F59E0B;
    --emerald:  #10B981;
    --red:      #EF4444;
    --slate:    #94A3B8;
    --white:    #E2E8F0;
    --r:        12px;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif;
    color: var(--white);
}
[data-testid="stHeader"]        { background: transparent !important; }
[data-testid="stSidebar"]       { background: var(--surf) !important; }
section[data-testid="stMain"] > div { padding-top: 0 !important; }
#MainMenu, footer, header       { visibility: hidden; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; }

/* Tabs */
[data-baseweb="tab-list"] {
    background: var(--surf2) !important;
    border-radius: var(--r) !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    color: var(--slate) !important;
    border-radius: 8px !important;
    padding: 8px 22px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--indigo) !important;
    color: #fff !important;
}

/* Button */
[data-testid="stButton"] > button {
    background: var(--indigo) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 32px !important;
    transition: all .18s ease !important;
}
[data-testid="stButton"] > button:hover {
    background: var(--indigo2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,.4) !important;
}

/* TextArea */
[data-testid="stTextArea"] textarea {
    background: var(--surf2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    color: var(--white) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.2) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surf2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--r) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surf2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: var(--slate) !important; font-family:'Space Grotesk',sans-serif !important; }
[data-testid="stMetricValue"] { color: var(--white) !important; font-family:'Space Grotesk',sans-serif !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--surf2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:99px; }
code { font-family:'JetBrains Mono',monospace !important; font-size:13px !important; }
</style>
""", unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:44px 0 30px; text-align:center; border-bottom:1px solid #2A3550; margin-bottom:36px;">
    <div style="display:inline-flex; align-items:center; gap:8px; background:#131929;
        border:1px solid #2A3550; border-radius:99px; padding:5px 16px;
        font-family:'Space Grotesk',sans-serif; font-size:11px; color:#818CF8;
        letter-spacing:.1em; text-transform:uppercase; margin-bottom:18px;">
        🔍 &nbsp;AI-Plagiarism-Tool
    </div>
    <h1 style="font-family:'Space Grotesk',sans-serif; font-size:clamp(28px,4.5vw,50px);
        font-weight:700; color:#E2E8F0; margin:0 0 10px; letter-spacing:-.02em; line-height:1.1;">
        Detect AI-Generated Text<br>
        <span style="color:#6366F1;">with Perplexity Analysis</span>
    </h1>
    <p style="font-family:'Inter',sans-serif; color:#94A3B8; font-size:16px;
        max-width:520px; margin:0 auto; line-height:1.6;">
        100% local · no API keys · n-gram LM + 20 linguistic signals<br>
        Supports text, PDF, Word, TXT, and 20+ code file types.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Cached imports ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_analyzer():
    from utils.analyzer import analyze_text
    return analyze_text

@st.cache_resource(show_spinner=False)
def load_file_processor():
    from utils.file_handler import process_file
    return process_file


# ── Input area ────────────────────────────────────────────────────────────────
tab_text, tab_file = st.tabs(["✏️  Paste Text", "📎  Upload File"])

input_text = ""
file_label = ""

with tab_text:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    raw = st.text_area(
        "text_input", height=230,
        placeholder="Paste any text — essay, email, article, code, assignment…",
        label_visibility="collapsed",
    )
    wc = len(raw.split()) if raw.strip() else 0
    st.markdown(f"<p style='font-size:12px;color:#94A3B8;margin-top:2px;'>{wc} words</p>",
                unsafe_allow_html=True)
    c1, _ = st.columns([1, 4])
    with c1:
        go_text = st.button("Analyze →", key="go_text", use_container_width=True)
    if go_text:
        if raw.strip():
            input_text, file_label = raw.strip(), "Pasted Text"
        else:
            st.warning("Paste some text first.")

with tab_file:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;">
    """ + "".join(
        f'<span style="background:#1C2438;border:1px solid #2A3550;border-radius:6px;'
        f'padding:3px 10px;font-size:11px;color:#94A3B8;font-family:\'JetBrains Mono\',monospace;">{ext}</span>'
        for ext in [".pdf",".docx",".txt",".py",".js",".ts",".java",".c/.cpp",".go",".rs",".rb",".html",".sql",".md","+ more"]
    ) + "</div>", unsafe_allow_html=True)

    up = st.file_uploader(
        "upload", label_visibility="collapsed",
        type=["pdf","docx","txt","py","js","ts","jsx","tsx","java","c","cpp","cs",
              "go","rs","rb","php","swift","kt","html","css","json","sql","md","sh","yaml","yml"],
    )
    c2, _ = st.columns([1, 4])
    with c2:
        go_file = st.button("Analyze →", key="go_file", use_container_width=True)
    if go_file:
        if up:
            try:
                pf = load_file_processor()
                txt, ftype = pf(up.name, up.read())
                if txt.strip():
                    input_text, file_label = txt.strip(), f"{ftype}: {up.name}"
                else:
                    st.error("No text could be extracted from this file.")
            except ValueError as e:
                st.error(str(e))
        else:
            st.warning("Upload a file first.")


# ── Results ───────────────────────────────────────────────────────────────────
if input_text:
    st.markdown("<hr style='border:none;border-top:1px solid #2A3550;margin:32px 0;'>",
                unsafe_allow_html=True)

    with st.spinner("Analyzing… running perplexity model and linguistic feature extraction"):
        fn = load_analyzer()
        R = fn(input_text)

    if "error" in R:
        st.error(R["error"])
        st.stop()

    cls   = R["classification"]
    prob  = R["ai_probability"]
    conf  = R["confidence"]
    summ  = R["summary"]
    ppl   = R["perplexity"]
    sigs  = R["signals"]
    flags = R["flagged_sentences"]
    stats = R["stats"]

    CLS_STYLE = {
        "AI Generated":      ("#EF4444", "#2A0A0A"),
        "Human Written":     ("#10B981", "#0A2218"),
        "Mixed / Uncertain": ("#F59E0B", "#261A07"),
    }
    cc, cbg = CLS_STYLE.get(cls, ("#94A3B8", "#1C2438"))

    # ── Verdict banner ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{cbg};border:1px solid {cc};border-radius:16px;
        padding:26px 30px;display:flex;align-items:center;
        justify-content:space-between;flex-wrap:wrap;gap:16px;margin-bottom:28px;">
        <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;
                color:{cc};letter-spacing:.09em;text-transform:uppercase;margin-bottom:6px;">
                Verdict
            </div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:30px;
                font-weight:700;color:{cc};">{cls}</div>
            <div style="font-family:'Inter',sans-serif;font-size:14px;
                color:#94A3B8;margin-top:8px;max-width:580px;line-height:1.55;">
                {html_module.escape(summ)}
            </div>
        </div>
        <div style="text-align:center;">
            <div style="width:90px;height:90px;border-radius:50%;border:3px solid {cc};
                display:flex;flex-direction:column;align-items:center;
                justify-content:center;background:{cbg};">
                <div style="font-family:'Space Grotesk',sans-serif;font-size:24px;
                    font-weight:700;color:{cc};">{conf}%</div>
                <div style="font-family:'Inter',sans-serif;font-size:10px;color:#94A3B8;">
                    confidence
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1: st.metric("AI Probability",    f"{round(prob*100,1)}%")
    with m2: st.metric("Perplexity",        f"{ppl['overall']:.1f}")
    with m3: st.metric("Human-likeness",    f"{ppl['normalized']:.0f}/100")
    with m4: st.metric("Words",             stats["word_count"])
    with m5: st.metric("Sentences",         stats["sentence_count"])
    with m6: st.metric("AI Phrases Found",  stats["ai_phrases_found"])

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Authorship bar ────────────────────────────────────────────────────
    ai_pct    = round(prob * 100, 1)
    human_pct = round(100 - ai_pct, 1)
    st.markdown(f"""
    <div style="background:#131929;border:1px solid #2A3550;border-radius:12px;
        padding:18px 22px;margin-bottom:22px;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;
            font-weight:600;color:#E2E8F0;margin-bottom:12px;">Authorship Distribution</div>
        <div style="display:flex;height:14px;border-radius:99px;overflow:hidden;margin-bottom:7px;">
            <div style="background:#EF4444;width:{ai_pct}%;"></div>
            <div style="background:#10B981;width:{human_pct}%;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="font-size:13px;color:#EF4444;font-family:'Inter',sans-serif;">
                AI Generated: {ai_pct}%</span>
            <span style="font-size:13px;color:#10B981;font-family:'Inter',sans-serif;">
                Human Written: {human_pct}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout: perplexity waveform | signals ──────────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("""<h3 style="font-family:'Space Grotesk',sans-serif;font-size:18px;
            font-weight:600;color:#E2E8F0;margin-bottom:14px;">
            📊 Sentence Perplexity Waveform</h3>""", unsafe_allow_html=True)

        sent_data = ppl.get("sentences", [])
        if sent_data:
            max_p = max(s["perplexity"] for s in sent_data) or 1
            bars = ""
            for s in sent_data[:25]:
                pct = min((s["perplexity"] / max_p) * 100, 100)
                if s["perplexity"] < 40:   bar_c = "#EF4444"
                elif s["perplexity"] < 100: bar_c = "#F59E0B"
                else:                       bar_c = "#10B981"
                preview = html_module.escape(s["text"][:68]) + ("…" if len(s["text"]) > 68 else "")
                bars += f"""
                <div style="margin-bottom:7px;" title="{html_module.escape(s['text'])}">
                    <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                        <span style="font-size:11px;color:#94A3B8;font-family:'Inter',sans-serif;
                            overflow:hidden;white-space:nowrap;text-overflow:ellipsis;
                            max-width:82%;">{preview}</span>
                        <span style="font-size:11px;color:{bar_c};font-family:'JetBrains Mono',monospace;
                            white-space:nowrap;margin-left:6px;">{s['perplexity']:.1f}</span>
                    </div>
                    <div style="background:#2A3550;border-radius:99px;height:5px;">
                        <div style="background:{bar_c};width:{pct:.1f}%;height:5px;border-radius:99px;"></div>
                    </div>
                </div>"""

            st.markdown(f"""
            <div style="background:#131929;border:1px solid #2A3550;border-radius:12px;
                padding:16px 18px;max-height:420px;overflow-y:auto;">
                <div style="display:flex;justify-content:space-between;margin-bottom:12px;">
                    <span style="font-size:12px;color:#94A3B8;font-family:'Space Grotesk',sans-serif;">
                        Overall: <strong style="color:#E2E8F0;">{ppl['overall']:.2f}</strong>
                        &nbsp;·&nbsp; Std dev: <strong style="color:#E2E8F0;">{ppl['ppl_std']:.2f}</strong>
                    </span>
                    <span style="font-size:11px;color:#94A3B8;font-style:italic;font-family:'Inter',sans-serif;">
                        {ppl['interpretation']}</span>
                </div>
                {bars}
            </div>
            <div style="display:flex;gap:16px;margin-top:8px;">
                <span style="font-size:11px;color:#EF4444;font-family:'Inter',sans-serif;">🔴 Low — AI-like</span>
                <span style="font-size:11px;color:#F59E0B;font-family:'Inter',sans-serif;">🟡 Medium</span>
                <span style="font-size:11px;color:#10B981;font-family:'Inter',sans-serif;">🟢 High — Human-like</span>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""<h3 style="font-family:'Space Grotesk',sans-serif;font-size:18px;
            font-weight:600;color:#E2E8F0;margin-bottom:14px;">
            🔬 Linguistic Signals</h3>""", unsafe_allow_html=True)

        w_colors = {"high": "#EF4444", "medium": "#F59E0B", "low": "#10B981"}
        dir_icons = {"ai": "🤖", "human": "🧑", "mixed": "🔀"}
        cards = ""
        for sig in sigs:
            wc2 = w_colors.get(sig["weight"], "#94A3B8")
            icon = dir_icons.get(sig["direction"], "")
            cards += f"""
            <div style="background:#131929;border:1px solid #2A3550;border-radius:9px;
                padding:11px 13px;margin-bottom:8px;">
                <div style="display:flex;align-items:center;gap:7px;margin-bottom:4px;">
                    <span style="width:7px;height:7px;border-radius:50%;background:{wc2};flex-shrink:0;"></span>
                    <span style="font-family:'Space Grotesk',sans-serif;font-size:12px;
                        font-weight:600;color:#E2E8F0;">{html_module.escape(sig['signal'])}</span>
                    <span style="margin-left:auto;font-size:10px;color:{wc2};
                        text-transform:uppercase;letter-spacing:.06em;">{sig['weight']}</span>
                    <span style="font-size:13px;">{icon}</span>
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:12px;color:#94A3B8;line-height:1.45;">
                    {html_module.escape(sig['observation'])}</div>
            </div>"""
        st.markdown(f"""
        <div style="max-height:430px;overflow-y:auto;padding-right:2px;">
            {cards}
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── Flagged sentences ─────────────────────────────────────────────────
    if flags:
        st.markdown("""<h3 style="font-family:'Space Grotesk',sans-serif;font-size:18px;
            font-weight:600;color:#E2E8F0;margin-bottom:12px;">
            🚩 Flagged Sentences</h3>""", unsafe_allow_html=True)
        for fs in flags:
            reasons_html = " · ".join(html_module.escape(r) for r in fs["reasons"])
            st.markdown(f"""
            <div style="background:#1A0A0A;border-left:3px solid #EF4444;
                border-radius:0 8px 8px 0;padding:12px 15px;margin-bottom:9px;">
                <div style="font-family:'Inter',sans-serif;font-size:13px;
                    color:#E2E8F0;margin-bottom:5px;line-height:1.5;">
                    &ldquo;{html_module.escape(fs['text'])}&rdquo;</div>
                <div style="font-family:'Inter',sans-serif;font-size:11px;color:#EF4444;">
                    ⚠️ {reasons_html} &nbsp;·&nbsp; perplexity: {fs['perplexity']}</div>
            </div>""", unsafe_allow_html=True)

    # ── Highlighted document ──────────────────────────────────────────────
    with st.expander("📄 Full Text with Highlights", expanded=False):
        safe = html_module.escape(input_text).replace("\n", "<br>")
        for fs in flags:
            orig = html_module.escape(fs["text"])
            if orig in safe:
                safe = safe.replace(orig,
                    f'<mark style="background:#3A1010;color:#EF9090;'
                    f'border-radius:3px;padding:0 3px;">{orig}</mark>', 1)
        st.markdown(f"""
        <div style="background:#131929;border:1px solid #2A3550;border-radius:12px;
            padding:18px 20px;font-family:'Inter',sans-serif;font-size:14px;
            color:#CBD5E1;line-height:1.8;max-height:380px;overflow-y:auto;">
            {safe}</div>""", unsafe_allow_html=True)
        if file_label:
            st.caption(f"Source: {file_label} · {stats['word_count']} words · "
                       f"{stats['char_count']} chars · TTR: {stats['ttr']:.3f} · "
                       f"Avg sentence: {stats['avg_sentence_length']} words")

    # ── Raw JSON ──────────────────────────────────────────────────────────
    with st.expander("🔧 Raw Analysis JSON", expanded=False):
        st.json(R)