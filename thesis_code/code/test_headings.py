# plagiotitloi 

from main import GreekSemanticChunker
from config import MODELS, DEFAULT_MODEL, SAMPLE_TEXT_PATH, SELECTED_MODEL
from read_pdf import extract_text_from_pdf
import requests
from config import API_BASE, MODEL_SETTINGS
import time
import psutil
import platform
from datetime import datetime
import subprocess
import GPUtil  # for NVIDIA GPU info
import evaluate
import csv
import matplotlib.pyplot as plt
import numpy as np
from evaluate_headings import evaluate_all



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
            gpu = gpus[0]  # Get primary GPU
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

def generate_title(summary: str, model: str, api_base: str, settings: dict) -> str:
    """Generate a short title in Greek based on the summary."""
    try:
        prompt = f"""Περίληψη: {summary}

Γράψε έναν σύντομο και περιεκτικό τίτλο για το παραπάνω κείμενο.
Ο τίτλος να είναι λιγότερος από 10 λέξεις, χωρίς εισαγωγικές φράσεις ή επεξηγήσεις."""

        response = requests.post(
            f"{api_base}/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                **settings
            }
        )

        if response.status_code == 200:
            return response.json()["response"].strip().replace('"', '')
        return "Άγνωστος Τίτλος"
    except Exception as e:
        print(f"Title generation error: {e}")
        return "Άγνωστος Τίτλος"


def test_headings():
    start_time = time.time()
    initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    # get initial GPU memory usage
    initial_gpu = get_gpu_info()
    
    chunker = GreekSemanticChunker()
    print(f"Testing with model: {chunker.model}")
    
    output_file = "output_headings.txt"
    
    try:
        pdf_path = "test-pdf.pdf"  # my PDF path - to allazw analoga to book

        # Extract paragraphs from the PDF
        paragraph_dict = extract_text_from_pdf(pdf_path)
        print(f"Successfully extracted {len(paragraph_dict)} paragraphs")
        

        # initialize the chunks list
        chunks = []

        # loop through paragraphs, chunk them and generate summaries & headings
        for i, para in enumerate(paragraph_dict.keys()):
            print(f"\nProcessing paragraph {i+1}/{len(paragraph_dict)}")
            try:
                summary = chunker._summarize_chunk(para)
                heading = generate_title(summary, chunker.model, API_BASE, MODEL_SETTINGS)
                chunk = {
                    "text": para,
                    "summary": summary,
                    "title": heading
                }
                chunks.append(chunk)
                print(f"Successfully processed chunk {i+1}")
            except Exception as e:
                print(f"Error processing paragraph {i+1}: {e}")
        
        end_time = time.time()
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        processing_time = end_time - start_time
        memory_used = final_memory - initial_memory
        
        system_info = get_system_info()

        # output system info and processing statistics
        with open(output_file, 'w', encoding='utf-8') as outfile:
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
            outfile.write(f"Model: {chunker.model}\n")
            outfile.write(f"Total Processing Time: {format_time(processing_time)}\n")
            outfile.write(f"Memory Usage: {memory_used:.2f}MB\n")
            outfile.write(f"Average Time per Chunk: {format_time(processing_time/len(chunks))}\n")
            outfile.write(f"Total Chunks: {len(chunks)}\n\n")

            outfile.write(f"{'='*80}\n\n")
            
            # write chunk information with headings
            for i, chunk in enumerate(chunks, 1):
                char_length = len(chunk['text'])
                token_length = count_tokens(chunk['text'])
                word_count = count_words(chunk['text'])
                
                print(f"\nChunk {i}:")
                print(f"Text: {chunk['text']}")
                print(f"Length: {char_length} characters, {word_count} words, {token_length} tokens")
                print(f"Summary: {chunk['summary']}")
                print(f"Heading: {chunk['title']}")
                
                outfile.write(f"{'='*80}\n")
                outfile.write(f"Chunk {i}\n")
                outfile.write(f"{'='*80}\n\n")
                outfile.write("Original Text:\n")
                outfile.write(f"{chunk['text']}\n\n")
                outfile.write(f"Length: {char_length} characters, {word_count} words, {token_length} tokens\n\n")
                outfile.write("Summary:\n")
                outfile.write(f"{chunk['summary']}\n\n")
                outfile.write("Heading:\n")
                outfile.write(f"{chunk['title']}\n\n")
            
            
            print("\n=== Ξεκινάει η πλήρης αξιολόγηση των αποτελεσμάτων ===\n")
            evaluate_all(chunks)

        

    except Exception as e:
        print(f"Error during test: {e}")




if __name__ == "__main__":
    test_headings()
