import fitz
import os


def extract_text_from_pdf(file_path):
    text = ""

    pdf = fitz.open(file_path)

    for page in pdf:
        text += page.get_text()

    pdf.close()

    return text


def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_text_from_md(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def parse_document(file_path):
    
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    elif extension == ".txt":
        return extract_text_from_txt(file_path)

    elif extension == ".md":
        return extract_text_from_md(file_path)

    else:
        return None