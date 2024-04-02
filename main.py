from text import extract_sentences
from LLM import LLM_Pipeline

pdf_path = ''
pipline = LLM_Pipeline()

with open(pdf_path, 'rb') as f:
    file = f.read()

    # Extract text
    sentances = extract_sentences(file)

    # Extract rules from paragraph 
    rules = pipline.classifier(sentances)

    # Convert to JSON from rules 
    JSON = pipline.json_from_string(rules)
    print(JSON)