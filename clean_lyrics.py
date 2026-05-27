import csv
import re
from pathlib import Path


INPUT_DIR = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")
CHUNKS_DIR = Path("data/chunks")

# Create output folders if they do not exist
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Find section headers (ex. [Chorus], [Bridge])
SECTION_PATTERN = r"(\[.*?\])"


def clean_lyrics(text):
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')

    text = re.sub(r"\[source[^\]]*\]", "", text, flags=re.IGNORECASE)

    text = re.sub(
        r"(?im)^\s*(you might also like|embed|\d+)\s*$",
        "",
        text
    )

    # Replace repeated spaces/tabs with one space
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_section_name(section):
    section = section.strip()
    section = re.sub(r":.*?(?=\])", "", section)

    return section


# Split song into chunks by section
# Each chunk contains: song title, album name, section name, lyrics
def chunk_song(song_title, album, lyrics):
    chunks = []

    # Split lyrics wherever a section header appears.
    # Because SECTION_PATTERN has parentheses, headers are kept
    parts = re.split(SECTION_PATTERN, lyrics)

    # Assume first block is Verse 1
    current_header = "[Verse 1]"

    for part in parts:
        part = part.strip()

        # Skip empty parts
        if not part:
            continue

        # If section header, update current header
        if part.startswith("[") and part.endswith("]"):
            current_header = clean_section_name(part)

        # Else, piece contains lyrics
        else:
            lines = [line.strip() for line in part.splitlines() if line.strip()]

            # If section contains only 1 line, drop it completely
            if len(lines) <= 1:
                continue

            chunk_data = {
                "song": song_title,
                "album": album,
                "section": current_header,
                "text": part
            }

            chunks.append(chunk_data)

    return chunks

# Process raw album file
# Output cleaned lyrics and chunks
def process_file(file_path):
    cleaned_output_path = CLEANED_DIR / file_path.name
    chunk_output_path = CHUNKS_DIR / f"{file_path.stem}_chunks.csv"

    with open(file_path, "r", encoding="utf-8", newline="") as infile, \
         open(cleaned_output_path, "w", encoding="utf-8", newline="") as cleaned_outfile, \
         open(chunk_output_path, "w", encoding="utf-8", newline="") as chunks_outfile:

        # Read raw input file as CSV rows:
        # Song title, album, lyrics
        reader = csv.reader(infile)

        cleaned_writer = csv.writer(cleaned_outfile)
        chunk_writer = csv.writer(chunks_outfile)

        # Write header row for chunk CSV
        chunk_writer.writerow([
            "song",
            "album",
            "section",
            "text"
        ])

        for row in reader:
            # Each row should have exactly: title, album, lyrics
            if len(row) != 3:
                print(f"Skipping weird row in {file_path.name}: {row[:2]}")
                continue

            title, album, lyrics = row

            title = title.strip()
            album = album.strip()
            cleaned_lyrics = clean_lyrics(lyrics)

            # Save the cleaned full-song version
            # This is useful for checking your cleaning output.
            cleaned_writer.writerow([
                title,
                album,
                "\n" + cleaned_lyrics
            ])

            chunks = chunk_song(
                title,
                album,
                cleaned_lyrics
            )

            # Save each chunk as one row in the chunk CSV
            for chunk in chunks:
                chunk_writer.writerow([
                    chunk["song"],
                    chunk["album"],
                    chunk["section"],
                    chunk["text"]
                ])

    print(f"Processed: {file_path.name}")


def main():
    # Process every .txt file in data/raw
    for file_path in INPUT_DIR.glob("*.txt"):
        process_file(file_path)


if __name__ == "__main__":
    main()