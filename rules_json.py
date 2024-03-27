import json
import base64

def convert_to_json(rules):
    """
    Convert extracted rules to JSON format.
    
    Args:
    - rules (list): List of extracted rules (strings)
    
    Returns:
    - json_data (str): JSON representation of the rules
    """
    formatted_rules = []
    for i, rule in enumerate(rules):
        formatted_rule = {
            f"expression{i+1}": {
                "expression": rule,
                "operator": "dummy_operator",
                "object1": "dummy_object1",
                "attribute1": "dummy_attribute1",
                "Value": "dummy_value",
                "Unit": "dummy_unit"
            }
        }
        formatted_rules.append(formatted_rule)
    
    json_data = json.dumps(formatted_rules, indent=4)
    return json_data

def download_json_file(json_data, filename="extracted_rules.json"):
    """
    Generate a download link for JSON file.
    
    Args:
    - json_data (str): JSON data to be downloaded
    - filename (str): Name of the JSON file
    
    Returns:
    - download_link (str): HTML download link for JSON file
    """
    b64 = base64.b64encode(json_data.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Download JSON File</a>'
    return href
