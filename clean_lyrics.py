import csv
import re
from pathlib import Path

INPUT_DIR = Path(".")
OUTPUT_DIR = Path("cleaned")
OUTPUT_DIR.mkdir(exist_ok=True)

def clean_lyrics(text):
    # Normalize curly quotes
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')

    # Remove source-style tags ONLY, not all bracket labels
    text = re.sub(r"\[source[^\]]*\]", "", text, flags=re.IGNORECASE)

    # Remove common junk lines if present
    text = re.sub(r"(?im)^\s*(you might also like|embed|\d+)\s*$", "", text)

    # Clean spaces/tabs but preserve lyric line breaks
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip each line
    text = "\n".join(line.strip() for line in text.splitlines())

    return text.strip()

def clean_file(file_path):
    output_path = OUTPUT_DIR / file_path.name

    with open(file_path, "r", encoding="utf-8", newline="") as infile, \
         open(output_path, "w", encoding="utf-8", newline="") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            if len(row) != 3:
                print(f"Skipping weird row in {file_path.name}: {row[:2]}")
                continue

            title, album, lyrics = row
            cleaned = clean_lyrics(lyrics)

            writer.writerow([title.strip(), album.strip(), "\n" + cleaned])

    print(f"Cleaned: {file_path} -> {output_path}")

for file_path in INPUT_DIR.glob("*.txt"):
    clean_file(file_path)