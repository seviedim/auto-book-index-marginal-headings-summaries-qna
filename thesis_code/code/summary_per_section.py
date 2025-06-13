# δημιουργω summary ανα ενοτητα του βιβλιου
import csv
import time
import requests
import platform
import psutil
from datetime import datetime
from statistics import mean
from extract_chapter_text import extract_chapter_chunks
import config

PDF_PATH = "PROS-ETAIREIES.pdf"
# olo to vivlio einai 20 sections - dokimazw ta 10 gia test
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

def summarize_text(text):
    model_name = config.DEFAULT_MODEL
    api_base = config.API_BASE.rstrip("/")
    url = f"{api_base}/generate"

    prompt = f"Δώσε μία σύντομη και περιεκτική περίληψη για το εξής κείμενο:\n\n{text}\n\nΠερίληψη:"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": config.MODEL_SETTINGS.get("options", {})
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        summary = data.get("response", "").strip()
    except Exception as e:
        print(f"Error summarizing: {e}")
        summary = "Error in summarization."

    return summary

if __name__ == "__main__":
    mem_before = psutil.virtual_memory().used / (1024 ** 2)
    start_time = time.time()
    
    all_section_chunks = extract_chapter_chunks(PDF_PATH, SECTIONS)
    summaries = []
    durations = []
    lengths = []
    errors = []

    for section, chunks in all_section_chunks.items():
        print(f"\nProcessing Section {section}...")
        section_start = time.time()
        full_text = "\n\n".join(chunks)

        if len(full_text) > 10000:
            print("Text too long, splitting summary in parts...")
            parts = [full_text[i:i+4000] for i in range(0, len(full_text), 4000)]
            summaries_in_parts = []

            for idx, part in enumerate(parts):
                print(f"Summarizing part {idx + 1}/{len(parts)}")
                summary_part = summarize_text(part)
                summaries_in_parts.append(summary_part)
                time.sleep(2)

            final_summary = "\n".join(summaries_in_parts)
        else:
            final_summary = summarize_text(full_text)

        duration = round(time.time() - section_start, 2)
        durations.append(duration)

        if "error" in final_summary.lower():
            errors.append(section)
        else:
            lengths.append(len(final_summary))

        summaries.append({"Section": section, "Summary": final_summary})
        time.sleep(1)

    # save summaries
    with open("section_summaries.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Section", "Summary"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(summaries)


    with open("section_summaries.txt", "w", encoding="utf-8") as file:
        for item in summaries:
            file.write(f"### {item['Section']} ###\n")
            file.write(item["Summary"] + "\n\n")

    # report
    total_time = round(time.time() - start_time, 2)
    mem_after = psutil.virtual_memory().used / (1024 ** 2)
    mem_delta = round(mem_after - mem_before, 2)

    now = datetime.now()
    platform_info = platform.platform()
    processor_info = platform.processor() or ""
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    mem_total = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    mem_available = round(psutil.virtual_memory().available / (1024 ** 3), 2)

    minutes = int(total_time // 60)
    seconds = round(total_time % 60, 2)
    avg_time = round(mean(durations), 2) if durations else 0

    report = f"""Processing Report - {now}
================================================================================\n\nSystem Information:
Platform: {platform_info}
Processor: {processor_info}
CPU Cores: {cpu_cores} (Physical), {cpu_threads} (Logical)
Total Memory: {mem_total}GB
Available Memory: {mem_available}GB\n\nProcessing Statistics:
Model: {config.DEFAULT_MODEL}
Total Processing Time: {minutes} minutes {seconds} seconds
Memory Usage: {mem_delta}MB
Average Time per Section: {avg_time} seconds
Total Sections: {len(SECTIONS)}
"""

    with open("processing_report.txt", "w", encoding="utf-8") as rep:
        rep.write(report)

    print("\nSummaries and report saved.")