from pathlib import Path
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Configuration & Paths ---
CHUNKS_DIR = Path("data/chunks")
OUTPUT_FILE = Path("data/song_profiles.json")
MODEL = "gpt-4o-mini"

openai_client = OpenAI()


def load_songs():
    """
    Group lyric chunks back into full songs so the profile is based on the
    song's overall narrative, not just one verse/chorus.
    """
    songs = {}

    for file in sorted(CHUNKS_DIR.glob("*_chunks.csv")):
        df = pd.read_csv(file)

        df = df.dropna(subset=["text"])
        df = df.drop_duplicates(subset=["album", "song", "section", "text"])

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


def generate_profile(song, album, lyrics):
    lyrics_text = "\n\n".join(lyrics)

    prompt = f"""
Create a concise emotional and narrative profile for this Taylor Swift song.

Song: {song}
Album: {album}

Lyrics:
{lyrics_text}

Important:
- Identify the narrator's role, not just the topic.
- Separate who is hurt from who caused the hurt.
- Separate active heartbreak from healed/reflected closure.
- This profile will be used to match songs to user situations, so POV and timeline matter.

Return JSON only in this exact format:
{{
  "song": "{song}",
  "album": "{album}",
  "themes": ["theme1", "theme2", "theme3"],
  "moods": ["mood1", "mood2"],
  "perspective": "short phrase describing narrator perspective",

  "speaker_role": "betrayed partner | cheating partner | regretful ex | person leaving | person left behind | hopeful partner | conflicted narrator | observer | other",

  "narrator_agency": "initiator | recipient | mutual | observer | unclear",

  "narrator_hurt_status": "hurt_by_other | hurting_other | self_blame | mutual_hurt | not_hurt | unclear",

  "relationship_stage": "pre-breakup | breakup | post-breakup | reconciliation | situationship | other",

  "timeline_state": "before_event | during_crisis | immediate_aftermath | unresolved_grief | healing | moved_on | reflective_closure | unclear",

  "good_for_user_roles": ["user role this song fits well"],
  "bad_for_user_roles": ["user role this song does NOT fit"],

  "situations": ["situation1", "situation2", "situation3"],

  "summary": "1 sentence summary of what emotional situation this song fits"
}}
"""

    response = openai_client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
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
        temperature=0.1
    )

    return json.loads(response.choices[0].message.content)


def main():
    songs = load_songs()
    profiles = {}

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            profiles = json.load(f)

    print(f"Found {len(songs)} songs.")

    for key, data in songs.items():
        # Use timeline_state as the schema-version marker.
        # If this exists, the profile is already using the newer POV/timeline schema.
        if key in profiles and "timeline_state" in profiles[key]:
            print(f"Skipping existing profile: {data['song']}")
            continue

        print(f"Generating profile: {data['song']}")

        try:
            profile = generate_profile(
                song=data["song"],
                album=data["album"],
                lyrics=data["lyrics"]
            )

            profiles[key] = profile

            # Save after every song so progress survives crashes/API errors.
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(profiles, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error generating profile for {data['song']}: {e}")
            print("Stopping so you can rerun safely after fixing the issue.")
            break

    print(f"Saved profiles to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()