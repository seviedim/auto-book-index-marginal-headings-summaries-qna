# gia na kanw extract polles selides (8)
import PyPDF2
import re

def remove_footer_notes(text: str) -> str:
    """Remove all text after first numbered footer line."""
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        if line.strip().replace(' ', '').startswith(tuple(f"{i}." for i in range(1, 10))):
            break
        clean_lines.append(line)
    return '\n'.join(clean_lines)

def extract_chapter_chunks(pdf_path, chapters: dict) -> dict:
    """
    Extract cleaned text chunks from chapters, based on page declarations.
    
    Each chapter is broken into text chunks of up to 8 pages.
    Cleaning steps include:
        - Joining hyphenated words
        - Removing footnotes
        - Flattening line breaks
    Page mapping is done by offsetting declared book pages to actual PDF pages.
    
    Args:
        pdf_path (str): Path to the PDF file
        chapters (dict): Dict like {1: [start_page, end_page], ...} using book page numbers
        
    Returns:
        dict: Dictionary with chapter numbers as keys and list of text chunks as values.
    """
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        chapter_text_chunks = {}

        for chapter_num, (book_start, book_end) in chapters.items():
            # convert book pages to actual PDF pages using offset
            start_pdf_page = book_start + 25
            end_pdf_page = book_end + 25

            chunks = []
            current_page = start_pdf_page

            while current_page <= end_pdf_page:
                chunk_end_page = min(current_page + 7, end_pdf_page)
                chunk_text = ""

                for page_num in range(current_page, chunk_end_page + 1):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()

                        if not text or len(text.strip()) < 50:
                            continue  # skip blank/short pages

                        text = re.sub(r'-\s*\n\s*', '', text)  # join hyphenated words
                        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
                        text = remove_footer_notes(text.strip())
                        text = text.replace('\n', ' ').strip()  # flatten line breaks

                        chunk_text += text + '\n'

                    except Exception as e:
                        print(f"Error processing page {page_num}: {e}")
                        continue

                if chunk_text.strip():
                    chunks.append(chunk_text.strip())

                current_page += 8

            chapter_text_chunks[chapter_num] = chunks

    return chapter_text_chunks

