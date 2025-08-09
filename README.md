# Book Indexing, Marginalia, and Q/A with Large Language Models

This project implements automated book processing with the following steps, using a Greek fine-tuned large language model.  
**This is my thesis project.**

## Model

Uses `meltemi_q8`, a quantized (8-bit) version of Meltemi-7B-Instruct v1.5, run locally via the Ollama API for privacy and autonomy.

## Workflow

1. Extract text from PDFs  
2. Create an index of key concepts  
3. Generate marginal headings per paragraph  
4. Summarize sections of the book  
5. Generate question/answer pairs per section  
6. Evaluate the quality of generated summaries and Q/A pairs  

## Outputs

Results are saved as CSV and TXT files.

## Requirements

- Python 3.10+  
- See `requirements.txt` for required packages  
- Ollama API installed and running locally  

## How to use

Run the scripts sequentially on your input PDFs to process the book following the steps above, including evaluation.

## Thesis PDF

You can access the complete thesis, including methodology, implementation details, and evaluation results, at the following link:  
[**Read Thesis (PDF)**](https://www.e-ce.uth.gr/research/theses-technical_reports/?lang=en)

