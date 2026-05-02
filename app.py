import streamlit as st
import requests
import base64

API_URL = "https://ai-plagiarism-detection-tool-5ew5.onrender.com"

st.set_page_config(page_title="AI & Plagiarism Detector", page_icon="🔍", layout="wide")

st.title("🔍 Pro AI-Based Plagiarism & Content Detector")
st.markdown("Advanced document analysis suite for detecting AI-generated text and structural plagiarism.")

# --- SIDEBAR & SETTINGS ---
st.sidebar.header("Options")
mode = st.sidebar.radio("Input Method", ["Text Input", "File Upload"])

st.sidebar.markdown("---")
st.sidebar.subheader("Reference Database")
st.sidebar.write("Upload a document to add it to the local plagiarism reference database.")
ref_file = st.sidebar.file_uploader("Upload Reference File", type=["txt", "pdf", "docx"], key="ref")

if st.sidebar.button("Add to Database"):
    if ref_file is not None:
        with st.spinner("Uploading..."):
            try:
                files = {"file": (ref_file.name, ref_file.getvalue(), ref_file.type)}
                r = requests.post(f"{API_URL}/upload-reference-file", files=files)
                if r.status_code == 200:
                    st.sidebar.success(r.json()["message"])
                else:
                    st.sidebar.error("Upload failed.")
            except Exception as e:
                st.sidebar.error("Could not connect to backend.")
    else:
        st.sidebar.warning("Select a file first.")

st.sidebar.info("Make sure the FastAPI backend is running! (`python -m uvicorn api:app --reload`)")

# --- UI LOGIC ---
def highlight_sentences(text, flagged_sentences):
    """Highlights plagiarized sentences in the text."""
    if not flagged_sentences:
        return text
        
    highlighted_text = text
    # Sort flagged sentences by length descending to avoid partial replacements messing up longer ones
    flagged_sentences = sorted(flagged_sentences, key=lambda x: len(x['sentence']), reverse=True)
    
    for fs in flagged_sentences:
        sentence = fs['sentence']
        source = fs['source']
        sim = fs['similarity']
        # Create a tooltip/highlight HTML block
        highlight_html = f'<span style="background-color: #ffcccc; color: #990000; padding: 2px 4px; border-radius: 3px; border-bottom: 2px solid red;" title="Source: {source} (Match: {sim}%)">{sentence}</span>'
        highlighted_text = highlighted_text.replace(sentence, highlight_html)
        
    return highlighted_text

def display_results(results_data, original_text=""):
    st.markdown("---")
    st.subheader("📊 Analysis Dashboard")
    
    # 1. METRICS ROW
    col1, col2, col3 = st.columns(3)
    
    ai_data = results_data.get("ai_analysis", {})
    ai_prob = ai_data.get("probability", 0.0)
    ai_class = ai_data.get("classification", "Unknown")
    
    plag_data = results_data.get("plagiarism_analysis", {})
    plag_score = plag_data.get("score", 0.0)
    matched_source = plag_data.get("matched_source", "None")
    flagged_sents = plag_data.get("flagged_sentences", [])
    
    with col1:
        st.metric(label="AI Generation Probability", value=f"{ai_prob}%", delta="AI Generated" if ai_class == "AI Generated" else "Human", delta_color="inverse" if ai_class == "AI Generated" else "normal")
    with col2:
        st.metric(label="Overall Plagiarism Score", value=f"{plag_score}%", delta=matched_source if plag_score > 0 else "Original", delta_color="inverse" if plag_score > 0 else "normal")
    with col3:
        st.metric(label="Sentences Flagged", value=len(flagged_sents))
        
    # 2. DETAILED BREAKDOWN
    st.markdown("### 📝 Text Analysis Breakdown")
    tab1, tab2 = st.tabs(["Highlighted Document", "Flagged Sources List"])
    
    with tab1:
        if original_text and flagged_sents:
            st.markdown("Sentences flagged for high similarity are highlighted in red. Hover over them to see the source.")
            st.markdown(f'<div style="line-height: 1.6; font-size: 16px; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">{highlight_sentences(original_text, flagged_sents)}</div>', unsafe_allow_html=True)
        elif original_text:
            st.success("No significant sentence-level plagiarism detected.")
            st.markdown(f'<div style="line-height: 1.6; font-size: 16px; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">{original_text}</div>', unsafe_allow_html=True)
        else:
            st.info("Upload a file or paste text to see the breakdown.")
            
    with tab2:
        if flagged_sents:
            st.write("The following specific sentences strongly matched local database references:")
            for fs in flagged_sents:
                st.error(f"**Matched Source:** `{fs['source']}` (Similarity: **{fs['similarity']}%**)\n\n\"{fs['sentence']}\"")
        else:
            st.success("No specific sentences flagged.")

    # 3. EXPORT
    st.markdown("---")
    with st.expander("Raw API JSON Data"):
        st.json(results_data)
        
    export_str = f"--- Analysis Report ---\n\nAI Probability: {ai_prob}%\nClassification: {ai_class}\n\nPlagiarism Score: {plag_score}%\nMatched Source: {matched_source}\nFlagged Sentences: {len(flagged_sents)}"
    b64 = base64.b64encode(export_str.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="analysis_report.txt">📄 Download Report as TXT</a>'
    st.markdown(href, unsafe_allow_html=True)

# --- MAIN INPUT AREA ---
if mode == "Text Input":
    text_input = st.text_area("Paste text here for analysis", height=200)
    if st.button("Analyze Text", type="primary"):
        with st.spinner("Analyzing text (Note: The AI Model is lazy-loaded and may take 20s on the very first run)..."):
            try:
                response = requests.post(f"{API_URL}/analyze-text", data={"text": text_input})
                if response.status_code == 200:
                    display_results(response.json(), original_text=text_input)
                else:
                    st.error(f"Error from backend: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to backend. Please ensure you are running `python -m uvicorn api:app --reload`.")
                
elif mode == "File Upload":
    uploaded_file = st.file_uploader("Upload a file to analyze (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
    if st.button("Analyze File", type="primary"):
        if uploaded_file is not None:
            with st.spinner("Analyzing file..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/upload-file", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        # We now get the full extracted text from the backend
                        st.success(f"File processed successfully: {data['filename']}")
                        display_results(data, original_text=data.get('extracted_text', ''))
                    else:
                        st.error(f"Error from backend: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to backend.")
        else:
             st.warning("Please upload a file first.")
