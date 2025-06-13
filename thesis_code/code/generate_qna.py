# kanw generate 2 QnA gia kathe section toy vivliou

import csv
import time
import psutil
import platform
import requests
import re
from datetime import datetime
from typing import Dict, List
from config import API_BASE, MODELS, DEFAULT_MODEL, MODEL_SETTINGS_V4

MAX_RETRIES = 2
DEBUG = True

def clean_summary_text(summary: str) -> str:
    lines = summary.strip().splitlines()
    filtered = [l for l in lines if not l.lower().startswith(("είσαι", "είμαι", "ο χρήστης", "ο ρόλος"))]
    return " ".join(filtered).replace('"', ' ').strip()

def build_prompt(cleaned: str) -> str:
    return (
        f"Περίληψη: {cleaned}\n\n"
        "Βάσει της παραπάνω περίληψης, γράψε δύο ερωτήσεις και τις απαντήσεις τους.\n"
        "Χρησιμοποίησε ΑΚΡΙΒΩΣ τη μορφή:\n\n"
        "Ερώτηση 1:\nΑπάντηση 1:\n\nΕρώτηση 2:\nΑπάντηση 2:"
    )

def parse_qna(raw: str) -> List[Dict[str, str]]:
    qna_pairs = []
    current_q = None
    current_a = None

    for line in raw.splitlines():
        line = line.strip()

        # skip irrelevant lines like certainty/confidence
        if line.lower().startswith(("βεβαιότητα:", "certainty:", "confidence:")):
            continue

        q_match = re.match(r"^(?:Q[12]|Ερώτηση(?:\s?\d*)?):\s*(.*)", line, re.IGNORECASE)
        a_match = re.match(r"^(?:A[12]|Απάντηση(?:\s?\d*)?):\s*(.*)", line, re.IGNORECASE)

        if q_match:
            if current_q and current_a:
                qna_pairs.append({"question": current_q, "answer": current_a})
                current_q, current_a = None, None
            current_q = q_match.group(1).strip()
        elif a_match:
            current_a = a_match.group(1).strip()

    if current_q and current_a:
        qna_pairs.append({"question": current_q, "answer": current_a})

    # deduplicate and return max 2
    unique_pairs = []
    for pair in qna_pairs:
        if pair not in unique_pairs:
            unique_pairs.append(pair)

    return unique_pairs[:2]


def generate_qna(summary: str, section: str) -> (List[Dict[str, str]], str):
    cleaned = clean_summary_text(summary)
    prompt = build_prompt(cleaned)
    options = {k: v for k, v in MODEL_SETTINGS_V4["options"].items() if v is not None}
    model = MODELS[DEFAULT_MODEL]["name"] + ":latest"

    raw = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{API_BASE}/generate",
                json={"model": model, "prompt": prompt, "stream": False, "options": options},
                timeout=600
            )
            if response.status_code != 200:
                print(f"[❌] API error on try {attempt} for {section}")
                continue

            raw = response.json().get("response", "")
            if DEBUG:
                print(f"\n[RAW {section} - Try {attempt}]\n{raw}\n")

            result = parse_qna(raw)
            if result:
                return result, raw
        except Exception as e:
            print(f"[❌] Exception in {section} try {attempt}: {e}")

    return [], raw

def generate_qna_from_csv(input_csv, output_csv, output_txt, report_file):
    start = time.time()
    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)

    with open(input_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("Summary")]

    with open(output_csv, "w", encoding="utf-8", newline="") as cfile, open(output_txt, "w", encoding="utf-8") as tfile:
        writer = csv.writer(cfile)
        writer.writerow(["Section", "Question1", "Answer1", "Question2", "Answer2", "RawQnA"])

        for r in rows:
            section, summary = r["Section"], r["Summary"]
            print(f"🔹 Section {section}...")
            qna, raw = generate_qna(summary, section)

            if not qna:
                print(f"⚠️ No usable QnA for {section}, raw saved.")
            elif len(qna) < 2:
                print(f"⚠️ Μόνο {len(qna)} ερώτηση/απάντηση για {section}")

            q1 = qna[0]["question"] if len(qna) > 0 else "—"
            a1 = qna[0]["answer"] if len(qna) > 0 else "—"
            q2 = qna[1]["question"] if len(qna) > 1 else "—"
            a2 = qna[1]["answer"] if len(qna) > 1 else "—"

            writer.writerow([section, q1, a1, q2, a2, raw.strip()])

            tfile.write(f"{'='*60}\nΕνότητα: {section}\n{'='*60}\n\n")
            tfile.write(f"❓ {q1}\n💬 {a1}\n\n❓ {q2}\n💬 {a2}\n\n")

    duration = time.time() - start
    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
    sysinfo = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_cores': psutil.cpu_count(logical=False),
        'cpu_threads': psutil.cpu_count(logical=True),
        'memory_total': f"{psutil.virtual_memory().total / (1024**3):.2f}GB",
        'memory_available': f"{psutil.virtual_memory().available / (1024**3):.2f}GB"
    }

    with open(report_file, "w", encoding="utf-8") as rep:
        rep.write(f"Processing Report - {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        rep.write("=" * 80 + "\n\n")
        for k, v in sysinfo.items():
            rep.write(f"{k}: {v}\n")
        rep.write(f"\nDuration: {int(duration//60)}m {duration%60:.1f}s\nMemory Used: {mem_after - mem_before:.2f}MB\n")

    print("\nΟλοκληρώθηκε! Αποθηκεύτηκαν:")
    print(f" - {output_csv}\n - {output_txt}\n - {report_file}")

if __name__ == "__main__":
    generate_qna_from_csv(
        "section_summaries.csv",
        "qna_from_summaries.csv",
        "qna_from_summaries.txt",
        "qna_report.txt"
    )
