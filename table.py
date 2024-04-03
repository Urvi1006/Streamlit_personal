import string
from collections import Counter
from itertools import count, tee
import cv2
import os
import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from paddleocr import PaddleOCR
import PyPDF2
import win32com.client
from docx.api import Document
import pandas as pd
word = win32com.client.Dispatch("Word.Application")
word.visible = 0

import random
import string

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    # print("Random string of length", length, "is:", result_str)
    return result_str

ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
def is_unflattened(pdf_file):
    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        for page in pdf_reader.pages:
            if '/Annots' in page:
                return True
        return False

table_detection_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection")

table_recognition_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition")


def PIL_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def pdf_to_images(pdf_path, output_folder_name='OUTPUTS_MASTER/output/images'):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_name, exist_ok=True)
    os.makedirs("OUTPUTS_MASTER\output\\tables\Output",exist_ok = True)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Create a list to store image paths
    image_paths = []

    # Iterate through each page in the PDF
    for page_number in range(pdf_document.page_count):
        # Get the page
        page = pdf_document[page_number]

        # Render the page as a pixmap
        pixmap = page.get_pixmap()

        # Convert the pixmap to a Pillow Image
        image = Image.frombytes(
            "RGB", [pixmap.width, pixmap.height], pixmap.samples)

        # Define the output image path
        output_image_path = os.path.join(
            output_folder_name, f"page_{page_number + 1}.png")

        # Save the image
        image.save(output_image_path)

        # Append the image path to the list
        image_paths.append(output_image_path)

    # Close the PDF document
    pdf_document.close()

    return image_paths


def cv_to_PIL(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def pytess(cell_pil_img, threshold: float = 0.5):
    cell_pil_img = TableExtractionPipeline.add_padding(
        pil_img=cell_pil_img, top=50, right=30, bottom=50, left=30, color=(255, 255, 255))
    result = ocr.ocr(np.asarray(cell_pil_img), cls=True)[0]

    text = ""
    if result != None:
        txts = [line[1][0] for line in result]
        text = " ".join(txts)
    return text


def sharpen_image(pil_img):

    img = PIL_to_cv(pil_img)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    pil_img = cv_to_PIL(sharpen)
    return pil_img


def uniquify(seq, suffs=count(1)):
    """Make all the items unique by adding a suffix (1, 2, etc).
    Credit: https://stackoverflow.com/questions/30650474/python-rename-duplicates-in-list-with-progressive-numbers-without-sorting-list
    `seq` is mutable sequence of strings.
    `suffs` is an optional alternative suffix iterable.
    """
    not_unique = [k for k, v in Counter(seq).items() if v > 1]

    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))
    for idx, s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            continue
        else:
            seq[idx] += suffix

    return seq


def binarizeBlur_image(pil_img):
    image = PIL_to_cv(pil_img)
    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]

    result = cv2.GaussianBlur(thresh, (5, 5), 0)
    result = 255 - result
    return cv_to_PIL(result)


def td_postprocess(pil_img):
    '''
    Removes gray background from tables
    '''
    img = PIL_to_cv(pil_img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 100),
                       (255, 5, 255))  # (0, 0, 100), (255, 5, 255)
    nzmask = cv2.inRange(hsv, (0, 0, 5),
                         (255, 255, 255))  # (0, 0, 5), (255, 255, 255))
    nzmask = cv2.erode(nzmask, np.ones((3, 3)))  # (3,3)
    mask = mask & nzmask

    new_img = img.copy()
    new_img[np.where(mask)] = 255

    return cv_to_PIL(new_img)


def table_detector(image, THRESHOLD_PROBA):
    '''
    Table detection using DEtect-object TRansformer pre-trained on 1 million tables

    '''

    feature_extractor = DetrImageProcessor(do_resize=True,
                                           size=800,
                                           max_size=800)
    encoding = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = table_detection_model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(
        outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

    return (probas[keep], bboxes_scaled)


def table_struct_recog(image, THRESHOLD_PROBA):
    '''
    Table structure recognition using DEtect-object TRansformer pre-trained on 1 million tables
    '''

    feature_extractor = DetrImageProcessor(do_resize=True,
                                           size=1000,
                                           max_size=1000)
    encoding = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = table_recognition_model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(
        outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

    return (probas[keep], bboxes_scaled)


class TableExtractionPipeline():

    colors = ["red", "blue", "green", "yellow", "orange", "violet"]

    @staticmethod
    def add_padding(pil_img,
                    top,
                    right,
                    bottom,
                    left,
                    color=(255, 255, 255)):
        '''
        Image padding as part of TSR pre-processing to prevent missing table edges
        '''
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    def sort_table_featuresv2(self, rows: dict, cols: dict):

        rows_ = {
            table_feature: (xmin, ymin, xmax, ymax)
            for table_feature, (
                xmin, ymin, xmax,
                ymax) in sorted(rows.items(), key=lambda tup: tup[1][1])
        }
        cols_ = {
            table_feature: (xmin, ymin, xmax, ymax)
            for table_feature, (
                xmin, ymin, xmax,
                ymax) in sorted(cols.items(), key=lambda tup: tup[1][0])
        }

        return rows_, cols_

    def individual_table_featuresv2(self, pil_img, rows: dict, cols: dict):

        for k, v in rows.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            rows[k] = xmin, ymin, xmax, ymax, cropped_img

        for k, v in cols.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cols[k] = xmin, ymin, xmax, ymax, cropped_img

        return rows, cols

    def crop_tables(self, pil_img, prob, boxes, delta_xmin, delta_ymin,
                    delta_xmax, delta_ymax):
        '''
        crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates 
        '''
        cropped_img_list = []

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

            xmin, ymin, xmax, ymax = xmin - delta_xmin, ymin - \
                delta_ymin, xmax + delta_xmax, ymax + delta_ymax
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cropped_img_list.append(cropped_img)

        return cropped_img_list

    def generate_structure(self,  model, pil_img, prob, boxes,
                           expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
        '''
        Co-ordinates are adjusted here by 3 'pixels'
        To plot table pillow image and the TSR bounding boxes on the table
        '''

        plt.figure(figsize=(32, 20))
        plt.imshow(pil_img)
        ax = plt.gca()
        rows = {}
        cols = {}
        idx = 0

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

            xmin, ymin, xmax, ymax = xmin, ymin, xmax, ymax
            cl = p.argmax()
            class_text = model.config.id2label[cl.item()]
            text = f'{class_text}: {p[cl]:0.2f}'
            # or (class_text == 'table column')
            if (class_text
                    == 'table row') or (class_text
                                        == 'table projected row header') or (
                                            class_text == 'table column'):
                ax.add_patch(
                    plt.Rectangle((xmin, ymin),
                                  xmax - xmin,
                                  ymax - ymin,
                                  fill=False,
                                  color=self.colors[cl.item()],
                                  linewidth=2))
                ax.text(xmin - 10,
                        ymin - 10,
                        text,
                        fontsize=5,
                        bbox=dict(facecolor='yellow', alpha=0.5))

            if class_text == 'table row':
                rows['table row.' +
                     str(idx)] = (xmin, ymin - expand_rowcol_bbox_top, xmax,
                                  ymax + expand_rowcol_bbox_bottom)
            if class_text == 'table column':
                cols['table column.' +
                     str(idx)] = (xmin, ymin - expand_rowcol_bbox_top, xmax,
                                  ymax + expand_rowcol_bbox_bottom)

            idx += 1

        return rows, cols

    def object_to_cellsv2(self, master_row: dict, cols: dict,
                          expand_rowcol_bbox_top, expand_rowcol_bbox_bottom,
                          padd_left):
        '''Removes redundant bbox for rows&columns and divides each row into cells from columns
        Args:

        Returns:


        '''
        cells_img = {}
        header_idx = 0
        row_idx = 0
        previous_xmax_col = 0
        new_cols = {}
        new_master_row = {}
        previous_ymin_row = 0
        new_cols = cols
        new_master_row = master_row

        for k_row, v_row in new_master_row.items():

            _, _, _, _, row_img = v_row
            xmax, ymax = row_img.size
            xa, ya, xb, yb = 0, 0, 0, ymax
            row_img_list = []

            for idx, kv in enumerate(new_cols.items()):
                k_col, v_col = kv
                xmin_col, _, xmax_col, _, col_img = v_col
                xmin_col, xmax_col = xmin_col - padd_left - 10, xmax_col - padd_left
                xa = xmin_col
                xb = xmax_col
                if idx == 0:
                    xa = 0
                if idx == len(new_cols) - 1:
                    xb = xmax
                xa, ya, xb, yb = xa, ya, xb, yb

                row_img_cropped = row_img.crop((xa, ya, xb, yb))
                row_img_list.append(row_img_cropped)

            cells_img[k_row + '.' + str(row_idx)] = row_img_list
            row_idx += 1

        return cells_img, len(new_cols), len(new_master_row) - 1

    def clean_dataframe(self, df):
        '''
        Remove irrelevant symbols that appear with tesseractOCR
        '''

        for col in df.columns:

            df[col] = df[col].str.replace("'", '', regex=True)
            df[col] = df[col].str.replace('"', '', regex=True)
            df[col] = df[col].str.replace(']', '', regex=True)
            df[col] = df[col].str.replace('[', '', regex=True)
            df[col] = df[col].str.replace('{', '', regex=True)
            df[col] = df[col].str.replace('}', '', regex=True)
        return df

    def create_dataframe(self, cell_ocr_res: list, max_cols: int,
                         max_rows: int):
        '''Create dataframe using list of cell values of the table, also checks for valid header of dataframe
        Args:
            cell_ocr_res: list of strings, each element representing a cell in a table
            max_cols, max_rows: number of columns and rows
        Returns:
            dataframe : final dataframe after all pre-processing 
        '''

        headers = cell_ocr_res[:max_cols]
        new_headers = uniquify(headers,
                               (f' {x!s}' for x in string.ascii_lowercase))
        counter = 0

        cells_list = cell_ocr_res[max_cols:]
        df = pd.DataFrame("", index=range(0, max_rows), columns=new_headers)

        cell_idx = 0
        for nrows in range(max_rows):
            for ncols in range(max_cols):
                df.iat[nrows, ncols] = str(cells_list[cell_idx])
                cell_idx += 1

        for x, col in zip(string.ascii_lowercase, new_headers):
            if f' {x!s}' == col:
                counter += 1
        header_char_count = [len(col) for col in new_headers]

        df = self.clean_dataframe(df)

        return df

    def start_process(self, image_path: str, TD_THRESHOLD=0.8, TSR_THRESHOLD=0.7,
                      OCR_THRESHOLD=0.5, padd_top=90, padd_left=40, padd_bottom=40,
                      padd_right=90,
                      delta_xmin=10,
                      delta_ymin=3,  # add offset to the bottom of the table
                      delta_xmax=10,  # add offset to the right of the table
                      delta_ymax=3,  # add offset to the top of the table
                      expand_rowcol_bbox_top=0,
                      expand_rowcol_bbox_bottom=0):
        '''
        Initiates process of generating pandas dataframes from raw pdf-page images

        '''
        image = Image.open(image_path).convert("RGB")
        probas, bboxes_scaled = table_detector(image,
                                               THRESHOLD_PROBA=TD_THRESHOLD)

        if bboxes_scaled.nelement() == 0:
            raise Exception("No table found in the pdf-page image")

        cropped_img_list = self.crop_tables(image, probas, bboxes_scaled,
                                            delta_xmin, delta_ymin, delta_xmax,
                                            delta_ymax)

        for idx, unpadded_table in enumerate(cropped_img_list):

            table = self.add_padding(unpadded_table, padd_top, padd_right,
                                     padd_bottom, padd_left)

            probas, bboxes_scaled = table_struct_recog(
                table, THRESHOLD_PROBA=TSR_THRESHOLD)
            rows, cols = self.generate_structure(table_recognition_model,
                                                 table, probas, bboxes_scaled,
                                                 expand_rowcol_bbox_top,
                                                 expand_rowcol_bbox_bottom)
            # st.write(len(rows), len(cols))
            rows, cols = self.sort_table_featuresv2(rows, cols)
            master_row, cols = self.individual_table_featuresv2(
                table, rows, cols)

            cells_img, max_cols, max_rows = self.object_to_cellsv2(
                master_row, cols, expand_rowcol_bbox_top,
                expand_rowcol_bbox_bottom, padd_left)

            sequential_cell_img_list = []
            for k, img_list in cells_img.items():
                for img in img_list:
                    # img = super_res(img)
                    # img = sharpen_image(img) # Test sharpen image next
                    # img = binarizeBlur_image(img)
                    # img = self.add_padding(img, 10,10,10,10)
                    # plt.imshow(img)
                    # c3.pyplot()
                    sequential_cell_img_list.append(
                        pytess(cell_pil_img=img, threshold=OCR_THRESHOLD))

            cell_ocr_res = sequential_cell_img_list
            # cell_ocr_res = asyncio.gather(*sequential_cell_img_list)

            df = self.create_dataframe(cell_ocr_res, max_cols, max_rows)

            return df

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     # Open the PDF file in binary mode
#     with open(pdf_path, 'rb') as file:
#         # Create a PdfFileReader object
#         pdf_reader = PyPDF2.PdfReader(file)
        
#         # Iterate through each page and extract text
#         for page_number in range(len(pdf_reader.pages)):
#             # Get the page object
#             page = pdf_reader.pages[page_number]
#             # Extract text from the page
#             text += page.extract_text()
            
#     return text
def extract_tables(pdf_path, output_folder_name='OUTPUTS_MASTER/output/images', output_csv_name='OUTPUTS_MASTER/output/tables'):
    # print("./data"+pdf_path.name,"<><><><><>>><><>><>><>>>><><")
    image_paths_list = pdf_to_images('./data/' + pdf_path.name, output_folder_name)
    te = TableExtractionPipeline()

    for img_path in image_paths_list:
        if "\\" in img_path:
            image_name = img_path.rsplit("/", 1)[-1]
        else:
            image_name = img_path

        print(f"Processing {image_name}")
        print(f"Processing {img_path}")

        try:
            df = te.start_process(img_path)
            df.to_csv(f'{output_csv_name}/{get_random_string(8)}.csv', index=False)
        except Exception as e:
            print(e)
            print(f"Error processing {image_name}")
            continue

def extract_tables_algo(pdf_path, output_folder_name):
    filename = os.path.basename(pdf_path)
    in_file = os.path.abspath(pdf_path)

    # Ensure the output folder exists
    output_folder = os.path.abspath(output_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    out_file = os.path.join(output_folder, filename[:-4] + ".docx")
    
    wb = word.Documents.Open(in_file)
    wb.SaveAs2(out_file, FileFormat=16)
    wb.Close()
    print("YOU ARE NOW USING THE ALGORITHMIC METHOD")

    print(f"Working on file {filename} ...")
    document = Document(out_file)
    tables = document.tables
    total_tables = 0
    for table_index, table in enumerate(tables, start=1):
        total_tables += 1

        table_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells] 
            table_data.append(row_data)

        df = pd.DataFrame(table_data)
        csv_file_path = os.path.join(f"{output_folder}", f"table_{get_random_string(8)}.csv")
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        df.to_csv(csv_file_path, index=False, header=False)

    print(f"Total number of tables extracted: {total_tables}")

def dual_pipeline(pdf_path):
    # # Extract text from the PDF
    # text = extract_text_from_pdf(pdf_path)
    # text = text.strip()

    # if len(text) > 100:
    #     print("Text length is greater than 100 characters. Using algorithmic extraction.")
    #     extract_tables_algo(pdf_path, output_folder_name="OUTPUTS_MASTER/Output")
    # else:
    #     print("Text length is less than or equal to 100 characters. Using regular extraction.")
        extract_tables(pdf_path, output_folder_name="OUTPUTS_MASTER/Output")

def dual_pipeline_2(pdf_path):
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    text = text.strip()

    if len(text) > 100:
        print("Text length is greater than 100 characters. Using algorithmic extraction.")
        extract_tables_algo(pdf_path, output_folder_name="OUTPUTS_MASTER/Output")
    else:
        print("Text length is less than or equal to 100 characters. Using regular extraction.")
        extract_tables(pdf_path, output_folder_name="OUTPUTS_MASTER/Output")

if __name__ == "__main__":
    
    # dual_pipeline('data\pdf\Basic Injection Molding Design Guidelines.pdf')
    dual_pipeline_2("Injection_Molding_Design_Guidelines_2017.pdf")
