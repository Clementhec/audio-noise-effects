import os
import re
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
from multiprocessing import Pool
from itertools import chain

from utils.logger import get_logger

class SoundBibleScraper:
    def __init__(self, base_url: str, download_dir: Path):
        self.sound_details = None
        self.base_url = base_url
        self.download_dir = download_dir
        self.logger = get_logger("SoundBibleScraper")

    @staticmethod
    def _fetch_page_hrefs(args):
        """Worker function to fetch hrefs from a single page."""
        page, base_url = args
        url = base_url.format(page)
        page_results = []

        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Failed to fetch {url}")
                return page_results

            soup = BeautifulSoup(response.text, "html.parser")

            # Look for <h3><a href=...>
            for h3 in soup.find_all("h3"):
                a_tag = h3.find("a", href=True)
                if a_tag:
                    page_results.append(
                        {"title": a_tag.text.strip(), "href": a_tag["href"]}
                    )
        except Exception as e:
            self.logger.info(f"Error fetching page {page}: {e}")

        return page_results

    @staticmethod
    def _download_sound_effect(
        url: str, output_path: Path, force_download: bool = False
    ) -> bool:
        """
        Download a sound effect file from URL.

        Args:
            url: URL to download from
            output_path: Local path to save file
            force_download: Re-download even if file exists

        Returns:
            True if successful, False otherwise
        """
        if output_path.exists() and not force_download:
            print(f"Already downloaded: {output_path.name}")
            return True

        try:
            self.logger.info(f"Downloading: {output_path.name}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            self.logger.info(f"Downloaded: {output_path.name}")
            return True

        except Exception as e:
            self.logger.info(f"Download failed: {e}")
            return False

    def fetch_sound_hrefs(self):
        # ? How do we know this ?
        N_PAGES = 237

        # Prepare arguments for multiprocessing
        page_args = [(page, self.base_url) for page in range(1, N_PAGES)]

        # Use multiprocessing pool with 4 workers
        with Pool(processes=4) as pool:
            results = pool.map(SoundBibleScraper._fetch_page_hrefs, page_args)

        # Flatten results from all pages
        all_results = list(chain.from_iterable(results))

        self.sound_details = pd.DataFrame(all_results)

    def fetch_sound_details_from_hrefs(self):
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            }
        )

        # initialise new columns
        self.sound_details["url"] = None
        self.sound_details["description"] = None
        self.sound_details["keywords"] = None
        self.sound_details["length"] = None

        for _, row in self.sound_details.iterrows():
            href = row["href"]
            title = row["title"]
            url = f"https://soundbible.com/{href}"

            try:
                response = session.get(url, timeout=12)
                if response.status_code != 200:
                    self.logger.info(f"Failed to fetch {url} (status {response.status_code})")
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

                row.update(
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
                self.logger.info(f"Error on {url}: {e}")

            # Avoid overloading website
            time.sleep(0.2)

    def clean_sounds_description(self):
        cleaned_descriptions = []

        for desc in self.sound_details["description"]:
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

        self.sound_details["clean_description"] = cleaned_descriptions

    def fetch_audio_urls_from_details(self):
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

        for idx, row in self.sound_details.iterrows():
            # get url or construct with href
            url = None
            if "url" in row and pd.notna(row["url"]):
                url = row["url"]
            elif "href" in row and pd.notna(row["href"]):
                url = urljoin(BASE_URL, str(row["href"]))
            else:
                self.logger.info(f"[{idx}] Aucun 'url' ni 'href' trouvé, skip")
                continue

            self.sound_details["audio_length"] = None
            self.sound_details["audio_file"] = None
            self.sound_details["audio_url"] = None

            try:
                resp = session.get(url, timeout=15)
                if resp.status_code != 200:
                    self.logger.info(f"status {resp.status_code}, skip")
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
                audio_length = None
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

                row.update(
                    {
                        "index": idx,
                        "url": url,
                        "audio_length": audio_length,
                        "audio_file": audio_file_name,
                        "audio_url": audio_url_full,
                    }
                )

            except Exception as e:
                self.logger.info(f"erreur sur {url}: {e}")

        self.sound_details["audio_url_wav"] = self.sound_details[
            "audio_url"
        ].str.replace("mp3", "wav")

    def download(self, sound_folder: Optional[Path] = None):
        """
        Download sound effects in the specified folder
        """
        if not sound_folder:
            sound_folder = self.download_dir
        self.sound_details["sound_location"] = self.sound_details["title"].apply(
            lambda t: sound_folder / Path(t.lower().replace(" ", "_") + ".wav")
        )
        sound_folder.mkdir(parents=True, exist_ok=True)
        sounds = self.sound_details[["title", "sound_location"]]
        sounds = [(s.sound_location, s.audio_url_wav) for s in sounds.itertuples()]
        with Pool(processes=-1) as pool:
            results = pool.map(SoundBibleScraper._download_sound_effect, sounds)
        n = sum([int(i) for i in results])
        self.logger.info(f"Downloaded {n} sounds out of {len(results)} ({n / len(results)}%)")

    def run(self) -> pd.DataFrame:
        self.fetch_sound_hrefs()
        self.fetch_sound_details_from_hrefs()
        self.clean_sounds_description()
        self.fetch_audio_urls_from_details()
        self.download()
        return self.sound_details


if __name__ == "__main__":
    BASE_URL = "https://soundbible.com/free-sound-effects-{}.html"
    download_dir = Path("data/sounds/soundbible")
    
    scraper = SoundBibleScraper(BASE_URL, download_dir)

    sound_details = scraper.run()

    sound_details.to_csv("data/sounds/soundbible_metadata.csv", index=False)
