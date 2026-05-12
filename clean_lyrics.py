import csv
import re
from pathlib import Path

INPUT_DIR = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")
CHUNKS_DIR = Path("data/chunks")

CLEANED_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

SECTION_PATTERN = r"(\[.*?\])"


def clean_lyrics(text):
    # Normalize curly quotes
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')

    # Remove source-style tags only
    text = re.sub(r"\[source[^\]]*\]", "", text, flags=re.IGNORECASE)

    # Remove common junk lines
    text = re.sub(
        r"(?im)^\s*(you might also like|embed|\d+)\s*$",
        "",
        text
    )

    # Clean spaces/tabs but preserve line breaks
    text = re.sub(r"[ \t]+", " ", text)

    # Strip each line
    text = "\n".join(line.strip() for line in text.splitlines())

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_section_name(section):
    section = section.strip()

    # Remove featured artist labels
    # Example:
    # [Verse 1: Future] -> [Verse 1]
    section = re.sub(r":.*?(?=\])", "", section)

    return section


def chunk_song(song_title, album, lyrics):
    chunks = []

    parts = re.split(SECTION_PATTERN, lyrics)

    # Many songs begin before a section tag appears
    current_header = "[Verse 1]"

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # Section header
        if part.startswith("[") and part.endswith("]"):
            current_header = clean_section_name(part)

        # Lyrics block
        else:
            chunk_data = {
                "song": song_title,
                "album": album,
                "section": current_header,
                "text": part
            }

            chunks.append(chunk_data)

    return chunks


def process_file(file_path):
    cleaned_output_path = CLEANED_DIR / file_path.name
    chunk_output_path = CHUNKS_DIR / f"{file_path.stem}_chunks.csv"

    with open(file_path, "r", encoding="utf-8", newline="") as infile, \
         open(cleaned_output_path, "w", encoding="utf-8", newline="") as cleaned_outfile, \
         open(chunk_output_path, "w", encoding="utf-8", newline="") as chunks_outfile:

        reader = csv.reader(infile)

        cleaned_writer = csv.writer(cleaned_outfile)
        chunk_writer = csv.writer(chunks_outfile)

        # Header row for chunk CSV
        chunk_writer.writerow([
            "song",
            "album",
            "section",
            "text"
        ])

        for row in reader:

            # Skip malformed rows
            if len(row) != 3:
                print(f"Skipping weird row in {file_path.name}: {row[:2]}")
                continue

            title, album, lyrics = row

            title = title.strip()
            album = album.strip()

            cleaned_lyrics = clean_lyrics(lyrics)

            # Save cleaned version
            cleaned_writer.writerow([
                title,
                album,
                "\n" + cleaned_lyrics
            ])

            # Create chunks
            chunks = chunk_song(
                title,
                album,
                cleaned_lyrics
            )

            # Save chunk rows
            for chunk in chunks:
                chunk_writer.writerow([
                    chunk["song"],
                    chunk["album"],
                    chunk["section"],
                    chunk["text"]
                ])

    print(f"Processed: {file_path.name}")


def main():
    for file_path in INPUT_DIR.glob("*.txt"):
        process_file(file_path)


if __name__ == "__main__":
    main()