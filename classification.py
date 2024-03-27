# text1.py

import random

def extract_rules_to_csv(extracted_text):
    """
    Dummy function to extract rules from extracted text and save them to a CSV file.
    This function returns a list of extracted rules.
    """
    # Dummy implementation: Extracting random sentences as rules
    num_rules = min(len(extracted_text), 10)  # Limit to 10 rules
    rules = random.sample(extracted_text, num_rules)
    
    return [str(rule) for rule in rules]  # Convert to string if not already

