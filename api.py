from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn
from io import BytesIO
import os
import shutil

from utils.file_handler import process_file
from utils.ai_detector import detect_ai_generated
from utils.plagiarism import check_plagiarism

app = FastAPI(title="AI & Plagiarism Detection API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI & Plagiarism Detection API. The server is running."}

@app.post("/analyze-text")
async def analyze_text(text: str = Form(...)):
    """
    Endpoint to analyze raw generated text
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
        
    ai_result = detect_ai_generated(text)
    plagiarism_result = check_plagiarism(text)
    
    return {
        "ai_analysis": ai_result,
        "plagiarism_analysis": plagiarism_result
    }

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to handle file uploads, extract text, and analyze it.
    """
    try:
        # Read the file directly into a BytesIO stream
        file_bytes = await file.read()
        file_stream = BytesIO(file_bytes)
        
        # Pass to file_handler
        extracted_text = process_file(file_stream, file.filename)
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract any text from the provided file.")
            
        # Run the detection engines
        ai_result = detect_ai_generated(extracted_text)
        plagiarism_result = check_plagiarism(extracted_text)
        
        return {
            "filename": file.filename,
            "extracted_text": extracted_text, 
            "ai_analysis": ai_result,
            "plagiarism_analysis": plagiarism_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@app.post("/upload-reference-file")
async def upload_reference_file(file: UploadFile = File(...)):
    """
    Endpoint to upload and save a file to the local database for future plagiarism reference.
    """
    try:
        db_path = "local_database"
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            
        file_location = os.path.join(db_path, file.filename)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        return {"message": f"Successfully added '{file.filename}' to reference database."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save reference file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
