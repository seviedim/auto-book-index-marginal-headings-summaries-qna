#titles_evaluation.py για πλαγιοτίτλους

import pandas as pd
import evaluate
from bert_score import score
import torch
import matplotlib.pyplot as plt
import json
import matplotlib.patches as mpatches

#εργαλεία evaluate
rouge = evaluate.load("rouge")

#gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_bertscore(candidate, reference):
    """Υπολογίζει BERTScore F1 ανάμεσα σε δύο κείμενα με xlm-roberta-large."""
    P, R, F1 = score(
        [candidate], [reference],
        model_type="xlm-roberta-large",
        lang="el",   #το xlm-roberta υποστηρίζει πολλές γλώσσες, συμπεριλαμβανομένων των Ελληνικών
        device=device,
        verbose=False
    )
    return F1[0].item()

def evaluate_all(chunks):
    """Πλήρης αξιολόγηση: περιλήψεις vs chunks και τίτλοι vs περιλήψεις."""
    results = []

    for idx, chunk in enumerate(chunks):
        original_text = chunk.get("text", "")
        summary = chunk.get("summary", "")
        title = chunk.get("title", "")

        if not summary or not title:
            continue

        #αξιολόγηση summary σε σχέση με chunk (chunk vs summary)
        F1_c = calculate_bertscore(summary, original_text)
        rouge_c = rouge.compute(predictions=[summary], references=[original_text])

        #αξιολόγηση title σε σχέση με summary (title vs summary)
        F1_t = calculate_bertscore(title, summary)
        rouge_t = rouge.compute(predictions=[title], references=[summary])

        results.append({
            'chunk': original_text,
            'summary': summary,
            'title': title,
            #Chunk vs Summary
            'bert_score_summary_f1': F1_c,
            'rouge1_summary': rouge_c['rouge1'],
            'rougeL_summary': rouge_c['rougeL'],
            #Title vs Summary
            'bert_score_title_f1': F1_t,
            'rouge1_title': rouge_t['rouge1'],
            'rougeL_title': rouge_t['rougeL'],
        })

    #αποθήκευση αποτελεσμάτων σε CSV
    df = pd.DataFrame(results)
    df.to_csv('titles_evaluation_full_results.csv', index=False, encoding='utf-8-sig')

    #υπολογισμός μο
    print("\n=== Μέσοι Όροι Αξιολόγησης ===")
    mean_scores = df.mean(numeric_only=True)
    print(mean_scores)

    #αποθήκευση μέσων όρων σε αρχείο txt
    with open('evaluation_summaries_and_titles.txt', 'w', encoding='utf-8') as f:
        f.write("=== Μέσοι Όροι Αξιολόγησης ===\n\n")
        for metric, value in mean_scores.items():
            f.write(f"{metric}: {value:.4f}\n")

    #BERTScore plot για τους τίτλους
    plot_scores(df['bert_score_title_f1'], "BERTScore (F1) Titles")

def plot_scores(scores, title):
    """Δημιουργεί bar plot για τα BERTScores με χρώματα ανά επίπεδο και labels."""

    #define colors based on score
    colors = []
    for score in scores:
        if score >= 0.82:
            colors.append('green')
        elif score < 0.8:
            colors.append('red')
        else:
            colors.append('blue')

    plt.figure(figsize=(12, 6))
    plt.bar([f"C{i+1}" for i in range(len(scores))], scores, color=colors, edgecolor='black')
    plt.xticks(rotation=45, fontsize=9)
    plt.xlabel("Chunks")
    plt.ylabel("BERTScore (F1)")
    plt.title(title)

    #add labels above bars
    for i, v in enumerate(scores):
        plt.text(i, v + 0.005, f"{v:.2f}", ha='center', va='bottom', fontsize=7,
                rotation=90,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))


    #add legend
    green_patch = mpatches.Patch(color='green', label='BERTScore >= 0.82 (Good)')
    blue_patch = mpatches.Patch(color='blue', label='BERTScore 0.80 - 0.82 (Moderate)')
    red_patch = mpatches.Patch(color='red', label='BERTScore < 0.80 (Low)')
    plt.legend(handles=[green_patch, blue_patch, red_patch])

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def main():
    with open('chunks_with_summaries_titles.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    evaluate_all(chunks)

if __name__ == "__main__":
    main()
