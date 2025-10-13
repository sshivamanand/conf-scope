import os
import csv
import json

# ---------- CONFIG ----------
DATA_DIR = "../data"  # folder containing acl, iclr, conll, rejected
CSV_PATH = "../data/conference_dataset.csv"

FOLDERS = {
    "rejected": {"iclr": 0, "conll": 0, "acl": 0},
    "iclr": {"iclr": 1, "conll": 0, "acl": 0},
    "conll": {"iclr": 0, "conll": 1, "acl": 0},
    "acl": {"iclr": 0, "conll": 0, "acl": 1},
}

def extract_intro_from_json(json_path):
    """
    Extract title and introduction from parsed PDF JSON.
    Returns empty strings if not available.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    title = data.get("metadata", {}).get("title", "")

    sections = data.get("metadata", {}).get("sections") or []
    if len(sections) > 1:
        intro = sections[1].get("text", "")
    else:
        intro = ""

    return title, intro


def create_csv_dataset():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    total_samples = 0

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["sl", "file_name", "title", "introduction", "iclr", "conll", "acl"])

        sl = 1
        for folder_name, labels in FOLDERS.items():
            folder_path = os.path.join(DATA_DIR, folder_name)
            if not os.path.exists(folder_path):
                print(f"⚠️ Folder not found: {folder_path}")
                continue

            json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
            for json_file in json_files:
                json_path = os.path.join(folder_path, json_file)
                title, intro = extract_intro_from_json(json_path)
                writer.writerow([
                    sl,
                    json_file,
                    title,
                    intro,
                    labels["iclr"],
                    labels["conll"],
                    labels["acl"]
                ])
                sl += 1
                total_samples += 1

                if total_samples % 100 == 0:
                    print(f"Processed {total_samples} papers...")

    print(f"\n✅ CSV dataset saved to {CSV_PATH}, total samples: {total_samples}")


if __name__ == "__main__":
    create_csv_dataset()
