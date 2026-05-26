from pathlib import Path
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Configuration & Paths ---
CHUNKS_DIR = Path("data/chunks")                 # Location of pre-processed CSV lyrics
OUTPUT_FILE = Path("data/song_profiles.json")    # Target file to store generated profiles
MODEL = "gpt-4o-mini"                            # OpenAI model optimized for structured JSON inference

openai_client = OpenAI()


# Group individual lyric chunks into a dictionary mapped by song to evaluate the tracks as whole units
def load_songs():
    songs = {}

    for file in sorted(CHUNKS_DIR.glob("*_chunks.csv")):
        df = pd.read_csv(file)
        
        # Data Cleaning: Drop rows missing text strings and handle duplicate row definitions
        df = df.dropna(subset=["text"])
        df = df.drop_duplicates(subset=["album", "song", "section", "text"])

        # Iterate through album spreadsheet and append lyrics to their respective song buckets
        for _, row in df.iterrows():
            profile_key = f"{row['album']}|||{row['song']}"

            if profile_key not in songs:
                songs[profile_key] = {
                    "album": row["album"],
                    "song": row["song"],
                    "lyrics": []
                }

            songs[profile_key]["lyrics"].append(str(row["text"]))

    return songs


# Interact with OpenAI API to generate structured semantic profile
def generate_profile(song, album, lyrics):
    lyrics_text = "\n\n".join(lyrics)

    prompt = f"""
Create a concise emotional and narrative profile for this Taylor Swift song.

Song: {song}
Album: {album}

Lyrics:
{lyrics_text}

Important:
- Identify not just the topic, but whose perspective the song is from.
- For example, distinguish betrayed partner vs cheating partner vs regretful ex vs person leaving.
- This profile will be used to match songs to user situations, so narrative POV matters.

Return JSON only in this exact format:
{{
  "song": "{song}",
  "album": "{album}",
  "themes": ["theme1", "theme2", "theme3"],
  "moods": ["mood1", "mood2"],
  "perspective": "short phrase describing narrator perspective",
  "speaker_role": "betrayed partner | cheating partner | regretful ex | person leaving | person left behind | hopeful partner | conflicted narrator | other",
  "relationship_stage": "pre-breakup | breakup | post-breakup | reconciliation | situationship | other",
  "situations": ["situation1", "situation2", "situation3"],
  "summary": "1 sentence summary of what emotional situation this song fits"
}}
"""

    response = openai_client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},  # Enforce strict JSON output schemas from LLM
        messages=[
            {
                "role": "system",
                "content": "You create concise song metadata for emotional and narrative retrieval. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2  # Keep deterministic focus to ensure uniform classification formatting
    )

    return json.loads(response.choices[0].message.content)


def main():
    songs = load_songs()
    profiles = {}

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint System: Load pre-existing data profiles to resume processing without re-billing APIs
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            profiles = json.load(f)

    print(f"Found {len(songs)} songs.")

    for key, data in songs.items():
        # Validate entry presence and schema updates to safely skip completed items
        if key in profiles and "speaker_role" in profiles[key]:
            print(f"Skipping existing profile: {data['song']}")
            continue

        print(f"Generating profile: {data['song']}")

        # Wrap response profile in fault-tolerant try-except block to guarantee incremental progress saves
        try:
            profile = generate_profile(
                song=data["song"],
                album=data["album"],
                lyrics=data["lyrics"]
            )
            profiles[key] = profile

            # Atomic Writing: Iteratively save profiles to disk to protect entries from crashes
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(profiles, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error generating profile for {data['song']}: {e}")
            print("Stopping so you can rerun safely after fixing the issue.")
            break

    print(f"Saved profiles to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()