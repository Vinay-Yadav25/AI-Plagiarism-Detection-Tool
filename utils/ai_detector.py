from transformers import pipeline

# Global pipeline instance, initialized only when first needed (Lazy Loading)
_ai_detector_pipe = None

def _get_detector():
    """
    Lazy loads the Hugging Face transformer model.
    This prevents the FastAPI server from hanging during startup.
    """
    global _ai_detector_pipe
    if _ai_detector_pipe is None:
        print("First request received! Loading AI Detection Model...")
        try:
            _ai_detector_pipe = pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta")
            print("AI Model loaded successfully into memory.")
        except Exception as e:
            print(f"Error loading AI detection model: {e}")
            raise e
    return _ai_detector_pipe

def detect_ai_generated(text: str) -> dict:
    """
    Analyzes text to characterize whether it is AI-generated or Human-written.
    Returns a dictionary with 'probability' and 'classification'.
    """
    if not text.strip():
        return {"probability": 0.0, "classification": "No text provided"}

    try:
        detector = _get_detector()
    except Exception as e:
        return {"probability": 0.0, "classification": "Error: Model failed to load"}

    # Transformers typically have a max sequence length (e.g., 512 tokens).
    # We truncate the text to ~2000 characters to prevent crashing.
    truncated_text = text[:2000]
    
    try:
        results = detector(truncated_text)
        # Typical output: [{'label': 'Human', 'score': 0.95}] or [{'label': 'ChatGPT', 'score': 0.99}]
        best_result = results[0]
        label = best_result['label']
        score = best_result['score']
        
        is_ai = False
        if label.lower() in ['chatgpt', 'fake']:
            is_ai = True
            
        probability_ai = score if is_ai else (1 - score)
        
        return {
            "probability": round(probability_ai * 100, 2),
            "classification": "AI Generated" if probability_ai > 0.50 else "Human Written"
        }
        
    except Exception as e:
        return {"probability": 0.0, "classification": f"Analysis Error: {str(e)}"}
