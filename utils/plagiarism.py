import os
import glob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.file_handler import process_file

DB_DIR = "local_database"

def _split_sentences(text: str) -> list:
    """Splits text into sentences using basic punctuation."""
    # Split on period, exclamation, or question mark followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out extremely short fragments
    return [s.strip() for s in sentences if len(s.strip()) > 15]

def load_database_documents():
    """
    Loads all reference documents from the local database.
    Returns a dict mapping filename to text content.
    """
    documents = {}
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        return documents
        
    for filepath in glob.glob(os.path.join(DB_DIR, '*')):
        if os.path.isfile(filepath):
            filename = os.path.basename(filepath)
            try:
                with open(filepath, 'rb') as f:
                     text = process_file(f, filename)
                     if text and text.strip():
                         documents[filename] = text
            except Exception as e:
                print(f"Failed to load DB file {filename}: {e}")
                
    return documents

def check_plagiarism(input_text: str) -> dict:
    """
    Advanced Plagiarism Checker (v2).
    Uses n-grams to catch reworded document structures.
    Also returns a breakdown of highly-similar sentences.
    """
    if not input_text.strip():
         return {"score": 0.0, "matched_source": "Empty text", "flagged_sentences": []}
         
    db_docs = load_database_documents()
    if not db_docs:
        return {"score": 0.0, "matched_source": "Local database is empty", "flagged_sentences": []}
        
    filenames = list(db_docs.keys())
    texts = list(db_docs.values())
    
    # --- 1. OVERALL DOCUMENT SIMILARITY ---
    doc_texts = texts + [input_text]
    
    # Use n-grams (1 to 3 words) to catch structural copying
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    try:
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
        # tfidf_matrix[-1] is the input text. tfidf_matrix[:-1] are DB texts.
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        
        max_sim_idx = cosine_similarities.argmax()
        max_sim_score = cosine_similarities[max_sim_idx]
        
        matched_source = filenames[max_sim_idx] if max_sim_score > 0.01 else None 
        overall_score = round(max_sim_score * 100, 2)
    except ValueError:
       # Can happen if the text is too short or only contains stop words
       overall_score = 0.0
       matched_source = None
       
    # --- 2. SENTENCE-LEVEL ANALYSIS ---
    flagged_sentences = []
    
    # Only run sentence analysis if there's a reason to suspect plagiarism
    if overall_score > 2.0:
        input_sentences = _split_sentences(input_text)
        
        db_sentences = []
        db_sentence_sources = []
        for fname, ftext in db_docs.items():
            sents = _split_sentences(ftext)
            db_sentences.extend(sents)
            db_sentence_sources.extend([fname] * len(sents))
            
        if db_sentences and input_sentences:
            try:
                # Vectorize at the sentence level (1 to 2 word ngrams is enough here)
                sent_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
                all_sents = db_sentences + input_sentences
                sent_matrix = sent_vectorizer.fit_transform(all_sents)
                
                db_matrix = sent_matrix[:len(db_sentences)]
                in_matrix = sent_matrix[len(db_sentences):]
                
                # Compute similarity of each input sentence against ALL db sentences
                sims = cosine_similarity(in_matrix, db_matrix)
                
                for i, sent_sims in enumerate(sims):
                    max_idx = sent_sims.argmax()
                    max_score = sent_sims[max_idx]
                    
                    # 40% similarity at sentence level is highly suspect
                    if max_score > 0.40: 
                        flagged_sentences.append({
                            "sentence": input_sentences[i],
                            "source": db_sentence_sources[max_idx],
                            "similarity": round(max_score * 100, 2)
                        })
            except Exception as e:
                print(f"Sentence analysis skipped/failed: {e}")

    return {
        "score": overall_score,
        "matched_source": matched_source if matched_source else "No significant match",
        "flagged_sentences": flagged_sentences
    }
