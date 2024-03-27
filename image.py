import fitz
import os
from PIL import Image

def extract_images(pdf_bytes, images_path):
    pdf_file = fitz.open("pdf", pdf_bytes)

    page_nums = len(pdf_file)

    images_list = []

    # Extract all images information from each page
    for page_num in range(page_nums):
        page_content = pdf_file[page_num]
        images_list.extend(page_content.get_images())

    # Save all the extracted images
    for i, img in enumerate(images_list, start=1):
        xref = img[0]
        base_image = pdf_file.extract_image(xref)
        image_bytes = base_image['image']
        image_ext = base_image['ext']
        image_name = str(i) + '.' + image_ext

        with open(os.path.join(images_path, image_name), 'wb') as image_file:
            image_file.write(image_bytes)

if __name__ == '__main__':
    extract_images('InjectionMoldedpart.pdf', 'Extracted_images')
