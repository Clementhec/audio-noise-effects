import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote


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


def fetch_audio_urls_from_details(sound_details: pd.DataFrame) -> pd.DataFrame:
    BASE_URL = "https://soundbible.com/"

    # Selectors
    SEL_AUDIO_PRIMARY = (
        "#ag1 > div:nth-child(2) > div > div > div > div.the-media > audio > source"
    )
    SEL_TIME_PRIMARY = "#ag1 > div:nth-child(2) > div > div > div > div.ap-controls.scrubbar-loaded > div.scrubbar > div.total-time"

    # Fallback selectors
    FALLBACK_AUDIO_SELECTORS = [
        "div.audioplayer-inner .the-media audio source",
        "div.audioplayer-inner audio source",
        "#ag1 source",
        "audio source",
    ]
    FALLBACK_TIME_SELECTORS = [
        "div.audioplayer-inner .total-time",
        ".total-time",
        "div.total-time",
    ]

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        }
    )

    results = []

    for idx, row in sound_details.iterrows():
        # get url or construct with href
        url = None
        if "url" in row and pd.notna(row["url"]):
            url = row["url"]
        elif "href" in row and pd.notna(row["href"]):
            url = urljoin(BASE_URL, str(row["href"]))
        else:
            print(f"[{idx}] Aucun 'url' ni 'href' trouvé, skip")
            results.append(
                {
                    "index": idx,
                    "url": None,
                    "audio_length": None,
                    "audio_file": None,
                    "audio_url": None,
                }
            )
            continue

        audio_file_name = None
        audio_url_full = None
        audio_length = None

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"  → status {resp.status_code}, skip")
                results.append(
                    {
                        "index": idx,
                        "url": url,
                        "audio_length": None,
                        "audio_file": None,
                        "audio_url": None,
                    }
                )
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # 1) Try main selector
            src = None
            tag = soup.select_one(SEL_AUDIO_PRIMARY)
            if tag and tag.has_attr("src"):
                src = tag["src"]
            else:
                # fallback - chercher un attribut data-source sur un parent (cas fréquent)
                # ex: <div ... data-source="mp3/airplane-takeoff_daniel_simion.mp3" ...>
                ds_tag = soup.select_one("#ag1 [data-source], [data-source]")
                if ds_tag and ds_tag.has_attr("data-source"):
                    data_src = ds_tag["data-source"]
                    # le contenu peut être JSON-like; on cherche le premier mp3/wav
                    m = re.search(
                        r"(?:mp3|wav)/[A-Za-z0-9._\-\s()%]+(?:\.mp3|\.wav)",
                        data_src,
                        re.IGNORECASE,
                    )
                    if m:
                        src = m.group(0)
                # si toujours rien, essayer autres fallback selectors
            if not src:
                for s in FALLBACK_AUDIO_SELECTORS:
                    t = soup.select_one(s)
                    if t and t.has_attr("src"):
                        src = t["src"]
                        break

            # Normalize src -> full url
            if src:
                src = src.strip()
                # si src est du type "mp3/xxx.mp3" (relatif) ou "./mp3/..." -> construire URL complète
                if src.startswith("http://") or src.startswith("https://"):
                    audio_url_full = src
                else:
                    # urljoin sur la page courante permet de couvrir ../ ou chemins relatifs
                    audio_url_full = urljoin(url, src)
                # extraire le nom de fichier décodé
                parsed = urlparse(audio_url_full)
                audio_file_name = unquote(os.path.basename(parsed.path))
            else:
                # dernier recours : parcourir tous les <source> et prendre le premier contenant mp3/wav
                found = None
                for s in soup.find_all("source"):
                    ssrc = s.get("src") or s.get("data-src") or ""
                    if re.search(r"\.(mp3|wav)$", ssrc, re.IGNORECASE):
                        found = ssrc
                        break
                if found:
                    audio_url_full = urljoin(url, found)
                    audio_file_name = unquote(
                        os.path.basename(urlparse(audio_url_full).path)
                    )

            # 2) audio length : main selector
            time_tag = soup.select_one(SEL_TIME_PRIMARY)
            if time_tag:
                audio_length = time_tag.get_text(strip=True)
            else:
                # fallback selectors
                for s in FALLBACK_TIME_SELECTORS:
                    t = soup.select_one(s)
                    if t and t.get_text(strip=True):
                        audio_length = t.get_text(strip=True)
                        break

            # autre fallback : chercher le premier div.total-time dans la page
            if not audio_length:
                t = soup.find("div", class_="total-time")
                if t:
                    audio_length = t.get_text(strip=True)

            results.append(
                {
                    "index": idx,
                    "url": url,
                    "audio_length": audio_length,
                    "audio_file": audio_file_name,
                    "audio_url": audio_url_full,
                }
            )

        except Exception as e:
            print(f"erreur sur {url}: {e}")
            results.append(
                {
                    "index": idx,
                    "url": url,
                    "audio_length": None,
                    "audio_file": None,
                    "audio_url": None,
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    base_url = "https://soundbible.com/free-sound-effects-{}.html"

    sound_hrefs = fetch_sound_hrefs(base_url)

    sound_hrefs.to_csv("soundbible_links.csv", index=False)

    sound_details = fetch_sound_details_from_hrefs(sound_hrefs)

    sound_details.to_csv("soundbible_details_from_section.csv", index=False)

    sound_details_clean = clean_sounds_description(sound_details)

    sound_details_clean.to_csv("soundbible_details_clean.csv", index=False)

    sound_audio_urls = fetch_audio_urls_from_details(sound_details)

    sound_audio_urls.to_csv("soundbible_audio_files.csv")
