# Book Indexing, Marginalities, and Q/A with Large Language Models

This project implements automated book indexing, marginality (heading) generation, and question/answering using a Greek fine-tuned large language model.

## Model

Uses `meltemi_q8`, a quantized (8-bit) version of Meltemi-7B-Instruct v1.5, run locally via Ollama API for privacy.

## What it does

- Extracts sections from PDFs
- Summarizes sections and creates headings
- Generates 2 question/answer pairs per section
- Saves results as CSV and TXT files

## Requirements

- Python 3.10+
- See `requirements.txt` for required packages
- Ollama API installed and running locally


## How to use

Run the Python scripts with your input PDFs.

