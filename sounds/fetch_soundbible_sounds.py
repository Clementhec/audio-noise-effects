import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup


def fetch_sound_hrefs(base_url: str):
    all_results = []
    for page in range(1, 237):  # Pages 1 to 20
        url = base_url.format(page)

        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Look for <h3><a href=...>
        for h3 in soup.find_all("h3"):
            a_tag = h3.find("a", href=True)
            if a_tag:
                all_results.append({"title": a_tag.text.strip(), "href": a_tag["href"]})

    return pd.DataFrame(all_results)


def fetch_sound_details_from_hrefs(sound_hrefs: pd.DataFrame) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        }
    )

    results = []

    for idx, row in sound_hrefs.iterrows():
        href = row["href"]
        title = row["title"]
        url = f"https://soundbible.com/{href}"

        try:
            response = session.get(url, timeout=12)
            if response.status_code != 200:
                print(f"Failed to fetch {url} (status {response.status_code})")
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            # --- Description depuis la section #course-details ---
            # On cible le premier <p> dans la colonne de gauche (col-lg-8)
            description = None
            section = soup.find("section", id="course-details")
            if section:
                p_tag = section.select_one(".col-lg-8 p")
                if not p_tag:
                    # fallback: premier <p> sous la section si la structure diffère
                    p_tag = section.find("p")
                if p_tag:
                    # Nettoyage des espaces / sauts de ligne
                    description = " ".join(p_tag.stripped_strings)

            # --- Keywords (on conserve la récupération via meta) ---
            kw_tag = soup.find("meta", {"name": "keywords"})
            keywords = (
                kw_tag["content"].split(",")
                if kw_tag and kw_tag.has_attr("content")
                else []
            )

            # --- Durée audio (dans la section, plus précis) ---
            time_tag = soup.select_one("#course-details .total-time")
            if not time_tag:
                # fallback global si besoin
                time_tag = soup.find("div", {"class": "total-time"})
            length = time_tag.get_text(strip=True) if time_tag else None

            results.append(
                {
                    "title": title,
                    "href": href,
                    "url": url,
                    "description": description,  # <-- maintenant depuis #course-details
                    "keywords": [k.strip() for k in keywords if k.strip()],
                    "length": length,
                }
            )

        except Exception as e:
            print(f"Error on {url}: {e}")

        # Politesse pour éviter de surcharger le site
        time.sleep(1)

    # DataFrame et export
    sounds_details = pd.DataFrame(results)
    return sounds_details


def clean_sounds_description(sound_details: pd.DataFrame) -> pd.DataFrame:
    cleaned_descriptions = []

    for desc in sound_details["description"]:
        if pd.isna(desc):
            cleaned_descriptions.append(None)
            continue

        pattern = r"[^.!?]*[Dd]ownload[^.!?]*[.!?]\s*"
        desc_filtered = re.sub(pattern, "", desc)

        # Then try to match the pattern
        match = re.search(r"Free\.\s*Get\s*(.*?)\s*in Wav or MP3", desc_filtered)
        if match:
            cleaned_descriptions.append(match.group(1).strip())
        else:
            cleaned_descriptions.append(desc_filtered)

    sound_details["clean_description"] = cleaned_descriptions
    return sound_details


if __name__ == "__main__":
    base_url = "https://soundbible.com/free-sound-effects-{}.html"

    sound_hrefs = fetch_sound_hrefs(base_url)

    sound_hrefs.to_csv("soundbible_links.csv", index=False)

    sound_details = fetch_sound_details_from_hrefs(sound_hrefs)

    sound_details.to_csv("soundbible_details_from_section.csv", index=False)

    sound_details_clean = clean_sounds_description(sound_details)

    sound_details_clean.to_csv("soundbible_details_clean.csv", index=False)
