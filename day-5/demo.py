import re
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup


def load_pdf_pypdf2(file_path):
    text = ""
    reader = PdfReader(file_path)

    for page in reader.pages:
        extracted = page.extract_text()
        text += extracted + "\n"
    return text

def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"Page\s*\d+", "", text)
    # text = re.sub(r"[ \t]+", " ", text) 
    # text = re.sub(r"\n\s*\n+", "\n", text)  

    text = re.sub(r"\s+", " ", text)
    return text.strip()

def save_clean_file(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def process_file(input_path, output_path):
    if input_path.endswith(".pdf"):
        raw = load_pdf_pypdf2(input_path)

    elif input_path.endswith(".txt"):
        raw = load_txt(input_path)

    else:
        raise ValueError("file must be in .pdf or .txt")
    
    
    cleaned = clean_text(raw)
    save_clean_file(cleaned, output_path)
    
    print("cleaning completed.")
    print("saved cleaned file to: cleaned_output.txt")


process_file("demo.txt", "cleaned_output.txt")

