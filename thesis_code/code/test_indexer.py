from main import GreekSemanticChunker
from config import MODELS, DEFAULT_MODEL, SAMPLE_TEXT_PATH, SELECTED_MODEL
from read_pdf import extract_text_from_pdf
import time
import psutil
import platform
from datetime import datetime
import subprocess
import GPUtil  # for NVIDIA GPU info

def count_tokens(text: str) -> int:
    """Rough estimation of tokens (words) in Greek text"""
    words = text.replace(',', ' ').replace('.', ' ').replace(';', ' ').split()
    return len(words)

def count_words(text: str) -> int:
    """Count actual words in Greek text"""
    clean_text = text.replace(',', ' ').replace('.', ' ').replace(';', ' ')
    words = [w for w in clean_text.split() if w.strip()]
    return len(words)

def get_gpu_info():
    """Get GPU information"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # get primary GPU
            return {
                'name': gpu.name,
                'memory_total': f"{gpu.memoryTotal}MB",
                'memory_used': f"{gpu.memoryUsed}MB",
                'memory_free': f"{gpu.memoryFree}MB",
                'gpu_load': f"{gpu.load*100:.1f}%",
                'gpu_temp': f"{gpu.temperature}°C"
            }
    except:
        # fallback to nvidia-smi if GPUtil fails
        try:
            output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
            return {'raw_info': output}
        except:
            return None
    return None

def get_system_info():
    """Get system hardware information"""
    gpu_info = get_gpu_info()
    
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_cores': psutil.cpu_count(logical=False),
        'cpu_threads': psutil.cpu_count(logical=True),
        'memory_total': f"{psutil.virtual_memory().total / (1024**3):.2f}GB",
        'memory_available': f"{psutil.virtual_memory().available / (1024**3):.2f}GB",
        'gpu': gpu_info
    }
    return info

def format_time(seconds):
    """Format time in minutes and seconds"""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes > 0:
        return f"{minutes} minutes {remaining_seconds:.2f} seconds"
    return f"{remaining_seconds:.2f} seconds"

def test_model():
    start_time = time.time()
    initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    # get initial GPU memory usage
    initial_gpu = get_gpu_info()
    
    chunker = GreekSemanticChunker()
    print(f"Testing with model: {chunker.model}")
    
    output_file = f"output_{SELECTED_MODEL.value}.txt"
    index_file = "index.txt"
    
    try:
        with open(SAMPLE_TEXT_PATH, 'r', encoding='utf-8') as file:
            test_text = file.read()
    except FileNotFoundError:
        print(f"Sample text file not found at: {SAMPLE_TEXT_PATH}")
        test_text = """
        Η Ελλάδα είναι μια χώρα με πλούσια ιστορία και πολιτισμό. 
        Από την αρχαιότητα μέχρι σήμερα, έχει συνεισφέρει σημαντικά 
        στην ανάπτυξη της φιλοσοφίας, της τέχνης και των επιστημών.
        """
    
    try:
        print("\nTesting semantic chunking...")

        pdf_path = "PROS-ETAIREIES.pdf"


        paragraph_dict = extract_text_from_pdf(pdf_path)
        print(f"Successfully extracted {len(paragraph_dict)} paragraphs")

        chunks, concepts_index = chunker.create_semantic_chunks(paragraph_dict)
    
        end_time = time.time()
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        processing_time = end_time - start_time
        memory_used = final_memory - initial_memory
        
        system_info = get_system_info()
        
        print(f"\nCreated {len(chunks)} chunks")
        
        with open(output_file, 'w', encoding='utf-8') as outfile, open(index_file, 'w', encoding='utf-8') as index_out:
            outfile.write(f"Processing Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            outfile.write(f"{'='*80}\n\n")
            
            outfile.write("System Information:\n")
            outfile.write(f"Platform: {system_info['platform']}\n")
            outfile.write(f"Processor: {system_info['processor']}\n")
            outfile.write(f"CPU Cores: {system_info['cpu_cores']} (Physical), {system_info['cpu_threads']} (Logical)\n")
            outfile.write(f"Total Memory: {system_info['memory_total']}\n")
            outfile.write(f"Available Memory: {system_info['memory_available']}\n")
            
            # write GPU information
            if system_info['gpu']:
                outfile.write("\nGPU Information:\n")
                if 'raw_info' in system_info['gpu']:
                    outfile.write(system_info['gpu']['raw_info'])
                else:
                    outfile.write(f"GPU: {system_info['gpu']['name']}\n")
                    outfile.write(f"GPU Memory Total: {system_info['gpu']['memory_total']}\n")
                    outfile.write(f"GPU Memory Used: {system_info['gpu']['memory_used']}\n")
                    outfile.write(f"GPU Memory Free: {system_info['gpu']['memory_free']}\n")
                    outfile.write(f"GPU Load: {system_info['gpu']['gpu_load']}\n")
                    outfile.write(f"GPU Temperature: {system_info['gpu']['gpu_temp']}\n")
            
            outfile.write("\nProcessing Statistics:\n")
            outfile.write(f"Model: {SELECTED_MODEL.value}\n")
            outfile.write(f"Total Processing Time: {format_time(processing_time)}\n")
            outfile.write(f"Memory Usage: {memory_used:.2f}MB\n")
            outfile.write(f"Average Time per Chunk: {format_time(processing_time/len(chunks))}\n")
            outfile.write(f"Total Chunks: {len(chunks)}\n\n")
            
            outfile.write(f"{'='*80}\n\n")
            
            # write chunk information
            for i, chunk in enumerate(chunks, 1):
                char_length = len(chunk['text'])
                token_length = count_tokens(chunk['text'])
                word_count = count_words(chunk['text'])
                
                print(f"\nChunk {i}:")
                print(f"Text: {chunk['text']}")
                print(f"Length: {char_length} characters, {word_count} words, {token_length} tokens")
                print(f"Summary: {chunk['summary']}")
                print(f"Key concepts: {', '.join(chunk['key_concepts'])}")
                
                outfile.write(f"{'='*80}\n")
                outfile.write(f"Chunk {i}\n")
                outfile.write(f"{'='*80}\n\n")
                outfile.write("Original Text:\n")
                outfile.write(f"{chunk['text']}\n\n")
                outfile.write(f"Length: {char_length} characters, {word_count} words, {token_length} tokens\n\n")
                outfile.write("Summary:\n")
                outfile.write(f"{chunk['summary']}\n\n")
                outfile.write("Key Concepts:\n")
                outfile.write(f"{', '.join(chunk['key_concepts'])}\n\n")
                
                

            # print key concepts indexer dict
            print(concepts_index)
            # sort by key concepts indexer dictionary alphabetically
            sorted_concepts = dict(sorted(concepts_index.items()))  
            print(sorted_concepts)
            outfile.write("\nKey Concepts Indexer:\n")
            outfile.write("="*15 + "\n\n")
            
            index_out.write("\nKey Concepts Indexer:\n")
            index_out.write("="*15 + "\n\n")

            # print the hierarchical index in the desired format to a file
            for main_word, pages in sorted(concepts_index.items()):
                # CASE 1: This is a single word concept (pages is a list of page numbers)
                if isinstance(pages, list):
                    # sort the page numbers and join them with commas
                    pages_str = ", ".join(map(str, sorted(pages)))
                    outfile.write(f"{main_word}: pages {pages_str}\n")
                
                # CASE 2: This is a first word with subcategories (pages is a dictionary)
                else:
                    # special case: if there's only one subcategory, print it as a whole concept
                    if len(pages) == 1:
                        # get the only subcategory
                        sub_word = list(pages.keys())[0]
                        sub_pages = pages[sub_word]
                        
                        # create the full concept by combining main_word and sub_word
                        full_concept = f"{main_word} {sub_word}"
                        
                        # sort the page numbers and join them with commas
                        pages_str = ", ".join(map(str, sorted(sub_pages)))
                        outfile.write(f"{full_concept}: pages {pages_str}\n")
                    
                    # regular case: multiple subcategories, print hierarchically
                    else:
                        # print the main category word first
                        outfile.write(f"{main_word}:\n")
                        # then iterate through all its subcategories (sorted alphabetically)
                        for sub_word, sub_pages in sorted(pages.items()):
                            # sort the page numbers and join them with commas
                            pages_str = ", ".join(map(str, sorted(sub_pages)))
                            # indent subcategories with a tab for better readability
                            outfile.write(f"\t{sub_word}: pages {pages_str}\n")

            print(f"\nResults have been written to {output_file} and {index_file}")
        
            
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    test_model()