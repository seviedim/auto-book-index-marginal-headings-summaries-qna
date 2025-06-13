# evaluation gia kathe summary per section

import csv
import json
import evaluate
import torch
from bert_score import score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from extract_chapter_text import extract_chapter_chunks
import config

# settings
PDF_PATH = "PROS-ETAIREIES.pdf"
CSV_PATH = "section_summaries.csv"

SECTIONS = {
    '§1': [3, 26],
    '§2': [27, 44],
    '§3': [45, 72],
    '§4': [75, 106],
    '§5': [107, 144],
    '§6': [145, 200],
    '§7': [201, 222],
    '§8': [223, 254],
    '§9': [255, 308],
    '§10': [309, 338],
    '§11': [339, 370],
    '§12': [373, 394],
    '§13': [395, 434],
    '§14': [435, 448],
    '§15': [449, 492],
    '§16': [493, 500],
    '§17': [501, 512],
    '§18': [513, 524],
    '§19': [525, 542],
    '§20': [543, 548]
}

# evaluate tools
rouge = evaluate.load("rouge")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_bertscore(candidate, reference):
    P, R, F1 = score(
        [candidate], [reference],
        model_type="xlm-roberta-large",
        lang="el",
        device=device,
        verbose=False
    )
    return F1[0].item()

def merge_summaries_with_texts():
    # fotwse summaries apo csv file
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        summaries = {row["Section"]: row["Summary"] for row in reader}

    # extract text
    all_chunks = extract_chapter_chunks(PDF_PATH, SECTIONS)

    # combine
    combined = []
    for section, chunks in all_chunks.items():
        original_text = "\n\n".join(chunks).strip()
        summary = summaries.get(section, "").strip()

        if not summary or not original_text:
            continue

        combined.append({
            "section": section,
            "text": original_text,
            "summary": summary
        })

    # save json
    with open("section_summaries.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print("Δημιουργήθηκε το section_summaries.json")
    return combined

# sunartisi evalauate_summaries :  bertscore - rouge
def evaluate_summaries(data):
    results = []

    for item in data:
        original = item["text"]
        summary = item["summary"]
        section = item["section"]

        bert_f1 = calculate_bertscore(summary, original)
        rouge_scores = rouge.compute(predictions=[summary], references=[original])

        results.append({
            'section': section,
            'bert_score_f1': bert_f1,
            'rouge1': rouge_scores['rouge1'],
            'rougeL': rouge_scores['rougeL']
        })

    df = pd.DataFrame(results)
    df.to_csv("summary_evaluation_results.csv", index=False, encoding="utf-8-sig")

    mean_scores = df.mean(numeric_only=True)
    with open("summary_evaluation_means.txt", "w", encoding="utf-8") as f:
        f.write("=== Μέσοι Όροι Αξιολόγησης ===\n\n")
        for metric, value in mean_scores.items():
            f.write(f"{metric}: {value:.4f}\n")

    print("\n=== Μέσοι Όροι ===")
    print(mean_scores)

    plot_scores(df['bert_score_f1'], "BERTScore (F1) Summaries")
    
    
# plots
def plot_scores(scores, title):
    colors = []
    for score in scores:
        if score >= 0.82:
            colors.append('green')
        elif score < 0.8:
            colors.append('red')
        else:
            colors.append('blue')

    plt.figure(figsize=(12, 6))
    plt.bar([f"S{i+1}" for i in range(len(scores))], scores, color=colors, edgecolor='black')
    plt.xticks(rotation=45, fontsize=9)
    plt.xlabel("Sections")
    plt.ylabel("BERTScore (F1)")
    plt.title(title)

    for i, v in enumerate(scores):
        plt.text(i, v + 0.015, f"{v:.2f}", ha='center', va='bottom', fontsize=7,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    green_patch = mpatches.Patch(color='green', label='≥ 0.82 (Good)')
    blue_patch = mpatches.Patch(color='blue', label='0.80 – 0.82 (Moderate)')
    red_patch = mpatches.Patch(color='red', label='< 0.80 (Low)')
    plt.legend(handles=[green_patch, blue_patch, red_patch])

    plt.tight_layout()
    plt.savefig("bert_score_summaries.png")
    plt.show()

if __name__ == "__main__":
    data = merge_summaries_with_texts()
    evaluate_summaries(data)
