import fitz
import re

def extract_sentences(pdf_bytes):
    doc = fitz.open("pdf", pdf_bytes)

    sentences = []
    
    for page in doc:
        text = page.get_text()
        # split text into sentences using regular expression
        sentences += re.split("(?<=[.!?])\s+", text)
    
    # remove extra whitespace and newlines
    sentences = [s.strip().replace('\n', '') for s in sentences]
    
    # remove empty sentences
    sentences = [s for s in sentences if s]
    return sentences


# if __name__ == "__main__":
#     pdf_path = ''
#     with open(pdf_path, 'rb') as f:
#         file = f.read()
#     sentances = extract_sentences(file)
#     print(sentances)