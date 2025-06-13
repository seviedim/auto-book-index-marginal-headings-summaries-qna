#code gia na kanw merge ta 2 csv: summaries - qnas

import csv

summaries_file = "section_summaries.csv"
qna_file = "qna_from_summaries.csv"
merged_file = "qna_with_summaries.csv"

section_to_summary = {}
with open(summaries_file, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        section = row["Section"].strip()
        summary = row["Summary"].strip()
        section_to_summary[section] = summary

with open(qna_file, encoding="utf-8") as infile, open(merged_file, "w", encoding="utf-8", newline="") as outfile:
    reader = csv.DictReader(infile)
    fieldnames = ["Section", "Summary", "Question1", "Answer1", "Question2", "Answer2", "RawQnA"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL) 
    writer.writeheader()

    for row in reader:
        section = row["Section"].strip()
        summary = section_to_summary.get(section, "")
        writer.writerow({
            "Section": section,
            "Summary": summary,
            "Question1": row["Question1"],
            "Answer1": row["Answer1"],
            "Question2": row["Question2"],
            "Answer2": row["Answer2"],
            "RawQnA": row["RawQnA"]
        })



print(f"Ολοκληρώθηκε. Δημιουργήθηκε το αρχείο: {merged_file}")
