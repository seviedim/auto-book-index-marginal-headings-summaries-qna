# evaluation script gia automati euretiriasi

import re
import json
import csv
import unicodedata
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict
import matplotlib.pyplot as plt
import pdfplumber
import difflib

# kanonikopoiei to text 
def normalize_text(text):
    """Normalize text: lowercase, remove accents, remove common Greek stopwords."""
    stopwords = {'του', 'της', 'των', 'και', 'με', 'σε', 'για', 'κατά', 'ως'}
    text = text.lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    words = text.split()
    filtered_words = [w for w in words if w not in stopwords]
    return ' '.join(filtered_words)

def load_manual_index(pdf_path):
    """Extract manual index terms from a book index PDF."""
    manual_terms = set()
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split('\n')
                for line in lines:
                    line = re.sub(r'\d+([,\sεπ\.]*\d*)*', '', line)
                    line = line.strip()
                    if line:
                        manual_terms.add(normalize_text(line))
    return manual_terms

def load_automated_index(txt_path):
    """Load the automated indexer output from a text file."""
    automated_terms = set()
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith('=') and not line.startswith('Key Concepts Indexer'):
                concept = line.split(':')[0].strip()
                if concept:
                    automated_terms.add(normalize_text(concept))
    return automated_terms

# fuzzy string matching me to difflib
def find_best_match(term, candidates, threshold=0.7):
    """Find the closest match for a term using difflib with lower threshold."""
    matches = difflib.get_close_matches(term, candidates, n=1, cutoff=threshold)
    return matches[0] if matches else None

def evaluate_indexes(manual_terms, automated_terms):
    """Compare the manual and automated indexes with partial match."""
    y_true = []
    y_pred = []

    matched_manual = set()
    matched_auto = set()

    for manual_term in manual_terms:
        match = find_best_match(manual_term, automated_terms)
        if match:
            matched_manual.add(manual_term)
            matched_auto.add(match)

    for term in manual_terms:
        y_true.append(1)
        y_pred.append(1 if term in matched_manual else 0)

    for term in automated_terms:
        if term not in matched_auto:
            y_true.append(0)
            y_pred.append(1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )

    return precision, recall, f1, matched_manual, matched_auto

#plots 

def plot_results(matched_manual, manual_terms, matched_auto, automated_terms):
    """Plot evaluation results and save the plot."""
    labels = ['Correct Matches', 'Missed (manual)', 'Wrong (auto)']
    sizes = [
        len(matched_manual),
        len(manual_terms) - len(matched_manual),
        len(automated_terms) - len(matched_auto)
    ]

    plt.figure(figsize=(8,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Indexer Evaluation Results')
    plt.axis('equal')
    plt.savefig("evaluation_index_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

#csv
def save_metrics_to_csv(precision, recall, f1, output_path="evaluation_metrics.csv"):
    """Save evaluation metrics to a CSV file."""
    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Precision', f"{precision:.4f}"])
        writer.writerow(['Recall', f"{recall:.4f}"])
        writer.writerow(['F1 Score', f"{f1:.4f}"])

def main():
    manual_index_pdf = "book_index.pdf"         # to ground-truth index (tou vivliou)
    automated_index_txt = "index.txt"            # to automated index output poy exw

    manual_terms = load_manual_index(manual_index_pdf)
    automated_terms = load_automated_index(automated_index_txt)

    print(f"Manual index terms: {len(manual_terms)}")
    print(f"Automated index terms: {len(automated_terms)}")

    precision, recall, f1, matched_manual, matched_auto = evaluate_indexes(manual_terms, automated_terms)

    print("\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    save_metrics_to_csv(precision, recall, f1)
    plot_results(matched_manual, manual_terms, matched_auto, automated_terms)

if __name__ == "__main__":
    main()
