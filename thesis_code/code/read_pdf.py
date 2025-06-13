import PyPDF2
import re

def clean_and_split_paragraphs(text: str) -> list:
    """
    Clean text and split into proper paragraphs by:
    1. Joining hyphenated words at line breaks
    2. Identifying true paragraph breaks
    
    Example:
        "παρότι γνωρίζει τι είναι η εταιρία, δύσκολα μπορεί να δώσει έναν πλήρη ορι-
         σμό της έννοιας.
         Συχνά η εταιρία περιγράφεται..."
         
    Becomes two paragraphs:
        ["παρότι γνωρίζει... ορισμό της έννοιας.",
         "Συχνά η εταιρία περιγράφεται..."]
    """
    # first join hyphenated words
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # now split on true paragraph breaks (sentence ending followed by newline)
    paragraphs = re.split(r'(?<=[.!?])\s*\n\s*(?=[Α-ΩΆΈΉΊΌΎΏ])', text)
    
    # clean up each paragraph
    paragraphs = [p.replace('\n', ' ').strip() for p in paragraphs if p.strip()]
    
    return paragraphs

# def remove_header(text: str) -> str:
#     """Remove headers from start of text, handling cases with or without newlines."""
    
#     # Handle continuous uppercase header at start
#     words = text.split()
#     header_length = 0
    
#     # Count length of uppercase words at start
#     for word in words:
#         if word.isupper():
#             header_length += len(word) + 1  # +1 for space
#         else:
#             break
    
#     if header_length > 0:
#         text = text[header_length:].strip()
    
#     # Then handle normal headers with newlines
#     lines = text.split('\n')
#     start_index = 0
    
#     for i, line in enumerate(lines[:3]):
#         if line.strip().isupper() or line.strip().startswith('§'):
#             start_index = i + 1
#         else:
#             break
            
#     return '\n'.join(lines[start_index:])

# def remove_header(text: str) -> str:
#    """
#    Remove header from text. Headers can be either:
#    1. All uppercase sentences (e.g. "ΓΕΝΙΚΗ ΕΙΣΑΓΩΓΗ ΣΤΟ ΕΤΑΙΡΙΚΟ ΔΙΚΑΙΟ")
#    2. Sections starting with § (e.g. "§ 1: Εταιρία: νομικός θεσμός...")
   
#    Examples:
#        "ΓΕΝΙΚΗ ΕΙΣΑΓΩΓΗ\nNormal text" -> "Normal text"
#        "§ 1: Section\nNormal text" -> "Normal text"
#    """
#    # Split into lines
#    lines = text.split('\n')
   
#    clean_lines = []
#    header_found = False
   
#    for line in lines:
#        # Skip empty lines
#        if not line.strip():
#            continue
           
#        # Check if line is a header
#        is_header = (
#            line.strip().isupper() or  # All uppercase
#            line.strip().startswith('§')  # Section symbol
#        )
       
#        # Once we find non-header text after header, start keeping lines
#        if header_found and not is_header:
#            clean_lines.append(line)
       
#        # Mark when we find a header
#        if is_header:
#            header_found = True
           
#    return '\n'.join(clean_lines)

# def remove_titles(text: str) -> str:
#    """
#    Remove title lines that start with:
#    1. Greek letters followed by dot (e.g., "Α. Title here")
#    2. Roman numerals followed by dot (e.g., "I. Title here")
   
#    Examples:
#        "Α. Title here\nNormal text" -> "Normal text"
#        "I. Title here\nNormal text" -> "Normal text"
#    """
#    lines = text.split('\n')
#    clean_lines = []
   
#    # Patterns for identifying titles
#    title_patterns = [
#        r'^[Α-Ω]\.',  # Greek capital letters
#        r'^[IVX]+\.',  # Roman numerals
#    ]
   
#    for line in lines:
#        # Skip empty lines
#        if not line.strip():
#            continue
           
#        # Check if line is a title
#        is_title = any(re.match(pattern, line.strip()) for pattern in title_patterns)
       
#        # Keep non-title lines
#        if not is_title:
#            clean_lines.append(line)
           
#    return '\n'.join(clean_lines)

# remove everything after the first numbered footer line is encountered.
def remove_footer_notes(text: str) -> str:
    """Remove all text after first numbered footer line."""
    
    # split into lines
    lines = text.split('\n')
    
    # keep lines until we find first footer
    clean_lines = []
    for line in lines:
        # Check if line starts with number + dot
        if line.strip().replace(' ', '').startswith(tuple(f"{i}." for i in range(1, 10))):
            break  # Stop when we find first footer
        clean_lines.append(line)
            
    return '\n'.join(clean_lines)


def extract_text_from_pdf(pdf_path):
    """Extract text as paragraphs with their corresponding page numbers."""
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        paragraph_dict = {}
        current_paragraph = []
        current_pages = []
        
        start_page = 29
        end_page = len(pdf_reader.pages)

        for page_num in range(start_page, end_page):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()


            # print(f"\nProcessing page {page_num}:")
            # print("First 100 chars:", text[:300])  # See start of each page's text
            
            if not text or len(text.strip()) < 50:
                print(f"Skipping page {page_num} - too short")
                continue
            
            # clean the text
            text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
            # text = remove_header(text.strip())
            # text = remove_titles(text.strip())
            text = remove_footer_notes(text.strip())

           

            # get paragraphs from this page
            paragraphs = clean_and_split_paragraphs(text)
            
            for para in paragraphs:
                if para.strip():
                    # check if this is continuation of previous paragraph
                    if current_paragraph and not current_paragraph[-1].strip().endswith('.'):
                        current_paragraph.append(para)
                        if page_num - 25 not in current_pages:
                            current_pages.append(page_num - 25)
                    else:
                        # if we have a complete paragraph, save it
                        if current_paragraph:
                            full_para = ' '.join(current_paragraph)
                            paragraph_dict[full_para] = current_pages
                        
                        # start new paragraph
                        current_paragraph = [para]
                        current_pages = [page_num - 25]
        
        # don't forget last paragraph
        if current_paragraph:
            full_para = ' '.join(current_paragraph)
            paragraph_dict[full_para] = current_pages
                
        return paragraph_dict

def save_extracted_text(paragraph_dict, output_path):
    """
    Save the extracted paragraphs and their page numbers to a text file.
    
    Args:
        paragraph_dict (dict): Dictionary with paragraphs as keys and page lists as values
        output_path (str): Path where to save the output file
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        for paragraph, pages in paragraph_dict.items():
            file.write(f"\n{'='*80}\n")
            file.write(f"Pages: {', '.join(map(str, pages))}\n")
            file.write(f"{'='*80}\n\n")
            file.write(paragraph)
            file.write('\n')

if __name__ == "__main__":
    pdf_path = "PROS-ETAIREIES.pdf"

    output_path = "extracted_text.txt"
    
    try:
        paragraph_dict = extract_text_from_pdf(pdf_path)
        save_extracted_text(paragraph_dict, output_path)
        print(f"Successfully extracted {len(paragraph_dict)} paragraphs")
    except Exception as e:
        print(f"An error occurred: {str(e)}")