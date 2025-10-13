from flask import Flask, jsonify, send_file
import os
import csv
import json
from datetime import datetime

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "..", "data")
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "output")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Mapping folder names to binary labels
FOLDER_BINARY_MAP = {
    "acl": "100",
    "iclr": "010",
    "other": "001",
    "rejected": "000"
}

@app.route('/')
def index():
    return jsonify({
        "message": f"Place your JSON subfolders inside '{DATA_FOLDER}'. Visit /parse to extract title, introduction, references, and folder labels."
    })

@app.route('/parse', methods=['GET'])
def parse_all_json():
    # Get all subfolders
    subfolders = [f for f in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, f))]
    json_entries = []

    for subfolder in subfolders:
        folder_path = os.path.join(DATA_FOLDER, subfolder)
        # Skip output folder
        if subfolder.lower() == "output":
            continue

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".json"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except Exception as e:
                        print(f"Skipping {file}: {e}")
                        continue

                    # Extract title safely
                    title = data.get("metadata", {}).get("title") or ""

                    # Extract introduction safely
                    introduction_text = ""
                    sections = data.get("metadata", {}).get("sections") or []  # <- ensures sections is list
                    for section in sections:
                        heading = section.get("heading") or ""
                        if "introduction" in heading.lower():
                            introduction_text = section.get("text") or ""
                            break
                    introduction_text = introduction_text.replace("\n", " ").strip()
                    introduction_text = introduction_text[:2000]  # truncate for readability

                    # Extract references safely
                    references = data.get("metadata", {}).get("references") or []
                    references_list = [ref.get("title", "") for ref in references]
                    references_str = " | ".join(references_list)  # pipe separated

                    # Folder labels
                    folder_label = subfolder.lower()
                    folder_label_binary = FOLDER_BINARY_MAP.get(folder_label, "001")  # default to 001

                    json_entries.append([
                        file,
                        title,
                        introduction_text,
                        references_str,
                        folder_label,
                        folder_label_binary
                    ])

    if not json_entries:
        return jsonify({"error": "No JSON files found in the subfolders"}), 404

    # Write CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_FOLDER, f"parsed_{timestamp}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "title", "introduction", "references", "folder_label", "folder_label_binary"])
        writer.writerows(json_entries)

    return send_file(csv_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
