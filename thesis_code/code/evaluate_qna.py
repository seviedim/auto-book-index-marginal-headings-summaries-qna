# kanw evaluate ta qna me vash 1. to arxiko keimeno 2. an oi apantiseis apantoun swsta tis erwtiseis

import csv
import requests
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from bert_score import score
from config import API_BASE, MODELS, DEFAULT_MODEL, MODEL_SETTINGS

QNA_INPUT_CSV = "qna_with_summaries.csv"
QNA_EVALUATION_CSV = "qna_evaluation.csv"
REPORT_FILE = "qna_evaluation_report.txt"

def calculate_bertscore(candidate, reference):
    candidate = candidate.strip()
    reference = reference.strip()
    if not reference:
        print(f"[⚠️] Empty summary for: {candidate}")
        return 0.0
    try:
        P, R, F1 = score([candidate], [reference], model_type="xlm-roberta-large", lang="el", verbose=False)
        return F1[0].item()
    except Exception as e:
        print(f"[⚠️] BERTScore error for '{candidate}': {e}")
        return 0.0

def check_answer_with_llm(question, answer):
    prompt = f"""Ερώτηση: {question}
Απάντηση: {answer}

Πες μου αν η απάντηση απαντά σωστά στην ερώτηση. Απάντησε μόνο με "Ναι" ή "Όχι" χωρίς εξηγήσεις."""
    try:
        response = requests.post(
            f"{API_BASE}/generate",
            json={
                "model": MODELS[DEFAULT_MODEL]["name"] + ":latest",
                "prompt": prompt,
                "stream": False,
                **MODEL_SETTINGS
            }
        )
        if response.status_code == 200:
            result = response.json().get("response", "").strip().lower()
            return "Ναι" if "ναι" in result else "Όχι"
        else:
            return "Άγνωστο"
    except Exception as e:
        print(f"Error checking answer: {e}")
        return "Σφάλμα"

def evaluate_qna():
    bert_scores = []
    correct_answers = 0
    total_answers = 0
    expected_total_qna = 0
    missing_qna = 0

    with open(QNA_INPUT_CSV, mode='r', encoding='utf-8') as infile, open(QNA_EVALUATION_CSV, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['Section', 'Question', 'Answer', 'BERTScore_Summary_Question', 'Answer_Correct']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            section = row['Section'].strip()
            summary = row.get('Summary', '').strip()
            expected_total_qna += 2

            for i in [1, 2]:
                q_col = f'Question{i}'
                a_col = f'Answer{i}'
                question = row.get(q_col, '').strip()
                answer = row.get(a_col, '').strip()

                if question and answer and question != "—" and answer != "—":
                    bertscore = calculate_bertscore(question, summary)
                    correctness = check_answer_with_llm(question, answer)
                    bert_scores.append(bertscore)
                    if correctness == "Ναι":
                        correct_answers += 1
                    total_answers += 1

                    writer.writerow({
                        'Section': section,
                        'Question': question,
                        'Answer': answer,
                        'BERTScore_Summary_Question': f"{bertscore:.4f}",
                        'Answer_Correct': correctness
                    })
                else:
                    missing_qna += 1
                    writer.writerow({
                        'Section': section,
                        'Question': question or '[MISSING]',
                        'Answer': answer or '[MISSING]',
                        'BERTScore_Summary_Question': 'N/A',
                        'Answer_Correct': 'Missing'
                    })

    avg_bertscore = np.mean(bert_scores) if bert_scores else 0
    accuracy = (correct_answers / total_answers) * 100 if total_answers else 0
    completeness = ((expected_total_qna - missing_qna) / expected_total_qna) * 100

    print("\nΗ αξιολόγηση QnA ολοκληρώθηκε! Αποθηκεύτηκαν:", QNA_EVALUATION_CSV)
    print(f"\nTotal Questions Evaluated: {total_answers}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Missing QnA: {missing_qna}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Completeness: {completeness:.2f}%")
    print(f"Average BERTScore: {avg_bertscore:.4f}")

    with open(REPORT_FILE, 'w', encoding='utf-8') as report:
        report.write("Evaluation Report\n=================\n")
        report.write(f"Total Questions Evaluated: {total_answers}\n")
        report.write(f"Correct Answers: {correct_answers}\n")
        report.write(f"Missing QnA: {missing_qna}\n")
        report.write(f"Expected Total QnA: {expected_total_qna}\n")
        report.write(f"Completeness: {completeness:.2f}%\n")
        report.write(f"Accuracy: {accuracy:.2f}%\n")
        report.write(f"Average BERTScore: {avg_bertscore:.4f}\n")

    #BERTScore per Question
    if bert_scores:
        colors = ['green' if s >= 0.82 else 'blue' if s >= 0.80 else 'red' for s in bert_scores]
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(bert_scores)), bert_scores, color=colors, edgecolor='black')
        plt.xticks(range(len(bert_scores)), [f"Q{i+1}" for i in range(len(bert_scores))], rotation=90, fontsize=8)
        plt.title('BERTScore per Question (vs Summary)')
        plt.xlabel('Question')
        plt.ylabel('BERTScore')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1)

        #score text on top of bars
        for i, s in enumerate(bert_scores):
            plt.text(i, s + 0.005, f"{s:.2f}", ha='center', va='bottom', fontsize=7, rotation=90)

        plt.legend(handles=[
            mpatches.Patch(color='green', label='>= 0.82 (Good)'),
            mpatches.Patch(color='blue', label='0.80 - 0.82 (Moderate)'),
            mpatches.Patch(color='red', label='< 0.80 (Low)')
        ])
        plt.tight_layout()
        plt.savefig("bertscore_per_question_summary.png")
        plt.show()

    #generation success pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(
        [expected_total_qna - missing_qna, missing_qna],
        labels=['Generated', 'Missing'],
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False
    )
    plt.title('QnA Generation Success')
    plt.savefig("qna_generation_success_piechart.png")
    plt.show()

    #answer evaluation distribution pie chart ---
    labels = ['Correct', 'Missing', 'Incorrect']
    sizes = [correct_answers, missing_qna, total_answers - correct_answers]

    #filter out zero values
    filtered_labels = [l for l, s in zip(labels, sizes) if s > 0]
    filtered_sizes = [s for s in sizes if s > 0]

    plt.figure(figsize=(6, 6))
    plt.pie(
        filtered_sizes,
        labels=filtered_labels,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False
    )
    plt.title('Answer Evaluation Distribution')
    plt.savefig("answer_accuracy_piechart.png")
    plt.show()

if __name__ == "__main__":
    evaluate_qna()
