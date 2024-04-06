from text import extract_sentences
from LLM import LLM_Pipeline

pdf_path = '/content/Streamlit_personal/data/Xometry Sheet Metal Design Guide 2020.pdf'
csv_path = './output.csv'
pipline = LLM_Pipeline()

with open(pdf_path, 'rb') as f:
    file = f.read()

    # Extract text
    sentances = extract_sentences(file)

    # Extract rules from paragraph 
    rules = pipline.classifier(sentances)
    print("RULES Start >>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(rules)
    print("RULES end   >>>>>>>>>>>>>>>>>>>>>>>>>>")

    # Convert to JSON from rules 
    JSON = pipline.json_from_list(rules, csv_path)
    print("JSON Start >>>>>>>>>>>>>>>>>>>>>>>")
    print(JSON)
    print("JSON end   >>>>>>>>>>>>>>>>>>>>>>>>")
