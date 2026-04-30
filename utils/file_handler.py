import pdfplumber
import docx

def extract_text_from_pdf(file_stream):
    text = ""
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text.strip()

def extract_text_from_docx(file_stream):
    text = ""
    try:
        doc = docx.Document(file_stream)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX: {e}")
    return text.strip()

def extract_text_from_txt(file_stream):
    try:
        # Assuming the stream is opened in binary mode, typical for FastAPI UploadFile
        return file_stream.read().decode('utf-8').strip()
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return ""

def process_file(file_obj, filename: str) -> str:
    """
    Extracts text from a file-like object based on its extension.
    """
    ext = filename.lower().split('.')[-1]
    if ext == 'pdf':
        return extract_text_from_pdf(file_obj)
    elif ext == 'docx':
        return extract_text_from_docx(file_obj)
    elif ext == 'txt':
        return extract_text_from_txt(file_obj)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
