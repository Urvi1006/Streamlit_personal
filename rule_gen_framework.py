import streamlit as st
import pandas as pd
from text import extract_sentences
from image import extract_images
from table import dual_pipeline
import os
from classification import extract_rules_to_csv  # Importing the extract_rules_to_csv function
from rules_json import convert_to_json, download_json_file  # Importing JSON conversion functions
import base64

# from highlighimport jsont import highlight_rules  # Importing the highlight_rules function

def get_rules_download_link(rules):
    csv = '\n'.join(rules)
    return f'<a href="data:file/csv;base64,{base64.b64encode(csv.encode()).decode()}" download="extracted_rules.csv">Download extracted rules as CSV</a>'

def main():
    st.title("PDF Data Extraction App")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select extraction option:", ["Upload", "Text", "Image", "Table"])

    if page == "Text":
        show_text_extraction()
    elif page == "Image":
        show_image_extraction()
    elif page == "Table":
        show_table_extraction()
    else:
        show_upload_page()

def show_upload_page():
    st.title("Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_file_uploader")

    if uploaded_file is not None:
        st.success("PDF file uploaded successfully!")
        if st.button("Next"):
            st.session_state.extracted_text = extract_uploaded_text(uploaded_file)
            st.session_state.page = "Text"
            st.session_state.initial_rules = extract_rules_to_csv(st.session_state.extracted_text)
            st.rerun()

def extract_uploaded_text(uploaded_file):
    pdf_bytes = uploaded_file.read()
    return extract_sentences(pdf_bytes)

def show_text_extraction():
    st.title("Text Extraction")
    if st.button("Back"):
        st.session_state.page = "Upload"
        st.rerun()

    extracted_text = st.session_state.extracted_text
    st.subheader("Extracted Text:")
    for sentence in extracted_text:
        st.write(sentence)

    if st.button("Next"):
        st.session_state.extracted_rules = extract_rules_to_csv(st.session_state.extracted_text)
        st.session_state.page = "Rules"
        st.rerun()

def show_extracted_rules():
    st.title("Extracted Rules")
    if st.button("Back"):
        st.session_state.page = "Text"  # Go back to the text extraction page
        st.rerun()

    extracted_rules = st.session_state.extracted_rules
    st.subheader("Extracted Rules:")
    for rule in extracted_rules:
        st.write(rule)

    st.markdown(get_rules_download_link(extracted_rules), unsafe_allow_html=True)

    # Offer the user to proceed to highlight rules in text
    if st.button("Next"):
        st.session_state.page = "Highlighted Text"
        st.rerun()

def show_highlighted_text():
    st.title("Highlighted Text with Rules")
    if st.button("Back"):
        st.session_state.page = "Rules"  # Go back to the extracted rules page
        st.rerun()

    extracted_text = st.session_state.extracted_text
    extracted_rules = st.session_state.extracted_rules

    st.subheader("Highlighted Text with Rules:")
    for sentence in extracted_text:
        highlighted_sentence = sentence
        for rule in extracted_rules:
            if rule in highlighted_sentence:
                highlighted_sentence = highlighted_sentence.replace(rule, f"<span style='color:red'>{rule}</span>")
        st.markdown(highlighted_sentence, unsafe_allow_html=True)

    # Offer the user to proceed to convert rules to JSON
    if st.button("Next"):
        st.session_state.page = "JSON Rules"
        st.rerun()

def show_json_rules():
    st.title("Extracted Rules (JSON Format)")
    if st.button("Back"):
        st.session_state.page = "Highlighted Text"  # Go back to the highlighted text page
        st.rerun()

    extracted_rules = st.session_state.extracted_rules
    json_data = convert_to_json(extracted_rules)
    st.code(json_data, language='json')
    st.markdown(download_json_file(json_data), unsafe_allow_html=True)

def show_image_extraction():
    st.title("Image Extraction")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_file_uploader")

    if uploaded_file is not None:
        st.success("PDF file uploaded successfully!")
        if st.button("Next"):
            extract_images_and_display(uploaded_file)

def extract_images_and_display(uploaded_file):
    pdf_bytes = uploaded_file.read()
    images_path = "Extracted_images"  # Set your desired path for extracted images

    # Remove existing images from the folder
    existing_images = os.listdir(images_path)
    for existing_image in existing_images:
        os.remove(os.path.join(images_path, existing_image))

    # Create the directory if it doesn't exist
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    extract_images(pdf_bytes, images_path)
    
    st.subheader("Extracted Images:")
    image_files = os.listdir(images_path)
    for image_file in image_files:
        st.image(os.path.join(images_path, image_file), caption=image_file, use_column_width=True)

def show_table_extraction():
    st.title("Table Extraction")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_file_uploader")

    if uploaded_file is not None:
        st.success("PDF file uploaded successfully!")
        if st.button("Next"):
            dual_pipeline(uploaded_file)
            # csv_files = [file if file.endswith(".csv") for file in os.listdir('OUTPUTS_MASTER/Output') ]
            csv_files = []
            for root, dirs, files in os.walk(os.path.abspath("OUTPUTS_MASTER/Output")):
                for file in files:
                    # print(os.path.join(root, file))
                    if file.endswith(".csv"):
                        csv_files.append(os.path.join(root, file))
            csv_dfs = [pd.read_csv(csv) for csv in csv_files]
            for df in csv_dfs:
                st.write(df)

if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "Upload"

    if st.session_state.page == "Upload":
        main()
    elif st.session_state.page == "Text":
        show_text_extraction()  # Display the extracted text page
    elif st.session_state.page == "Rules":
        show_extracted_rules()
    elif st.session_state.page == "Highlighted Text":
        show_highlighted_text()
    elif st.session_state.page == "JSON Rules":
        show_json_rules()
