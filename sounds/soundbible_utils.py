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

from utils.logger import setup_logger


class SoundBibleScraper:
    def __init__(self, base_url: str, download_dir: Path):
        self.sound_details = None
        self.base_url = base_url
        self.download_dir = download_dir
        self.logger = setup_logger(
            "SoundBibleScraper", log_file=Path("logs/soundbible_scraper.log")
        )

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
            print(f"Error fetching page {page}: {e}")

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
            print(f"Downloading: {output_path.name}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            print(f"Downloaded: {output_path.name}")
            return True

        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def fetch_sound_hrefs(self):
        # ? How do we know this ?
        N_PAGES = 237

        # Prepare arguments for multiprocessing
        page_args = [(page, self.base_url) for page in range(1, N_PAGES)]

        # Use multiprocessing pool with 4 workers
        self.logger.info("Fetch sounds hyperrefs...")
        with Pool(processes=4) as pool:
            results = pool.map(SoundBibleScraper._fetch_page_hrefs, page_args)

        # Flatten results from all pages
        all_results = list(chain.from_iterable(results))

        self.sound_details = pd.DataFrame(all_results)

    @staticmethod
    def _fetch_single_sound_detail(args):
        """Worker function to fetch details for a single sound."""
        index, href = args
        url = f"https://soundbible.com/{href}"

        result = {
            "index": index,
            "url": None,
            "description": None,
            "keywords": None,
            "length": None,
        }

        try:
            # Create session per worker
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                }
            )

            response = session.get(url, timeout=12)
            if response.status_code != 200:
                print(f"Failed to fetch {url} (status {response.status_code})")
                return result

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract description from course-details section
            description = None
            section = soup.find("section", id="course-details")
            if section:
                p_tag = section.select_one(".col-lg-8 p")
                if not p_tag:
                    p_tag = section.find("p")
                if p_tag:
                    description = " ".join(p_tag.stripped_strings)

            # Extract keywords from meta tag
            kw_tag = soup.find("meta", {"name": "keywords"})
            keywords = (
                kw_tag["content"].split(",")
                if kw_tag and kw_tag.has_attr("content")
                else []
            )

            # Extract audio length
            time_tag = soup.select_one("#course-details .total-time")
            if not time_tag:
                time_tag = soup.find("div", {"class": "total-time"})
            length = time_tag.get_text(strip=True) if time_tag else None

            result.update({
                "url": url,
                "description": description,
                "keywords": [k.strip() for k in keywords if k.strip()],
                "length": length,
            })

        except Exception as e:
            print(f"Error on {url}: {e}")

        # Small delay to avoid overloading
        time.sleep(0.05)
        return result

    def fetch_sound_details_from_hrefs(self):
        self.logger.info("Fetch sound details from hyperref...")

        # Prepare arguments: (index, href) tuples
        fetch_args = [(i, row["href"]) for i, row in self.sound_details.iterrows()]

        # Use multiprocessing pool with 8 workers
        with Pool(processes=8) as pool:
            results = pool.map(SoundBibleScraper._fetch_single_sound_detail, fetch_args)

        # Update DataFrame with results
        for result in results:
            idx = result["index"]
            self.sound_details.at[idx, "url"] = result["url"]
            self.sound_details.at[idx, "description"] = result["description"]
            self.sound_details.at[idx, "keywords"] = result["keywords"]
            self.sound_details.at[idx, "length"] = result["length"]

    def clean_sounds_description(self):
        cleaned_descriptions = []

        self.logger.info("Clean sound description...")
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

    @staticmethod
    def _fetch_single_audio_url(args):
        """Worker function to fetch audio URL for a single sound."""
        idx, url, href = args
        BASE_URL = "https://soundbible.com/"

        # Selectors
        SEL_AUDIO_PRIMARY = (
            "#ag1 > div:nth-child(2) > div > div > div > div.the-media > audio > source"
        )
        SEL_TIME_PRIMARY = "#ag1 > div:nth-child(2) > div > div > div > div.ap-controls.scrubbar-loaded > div.scrubbar > div.total-time"

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

        result = {
            "index": idx,
            "url": url,
            "audio_length": None,
            "audio_file": None,
            "audio_url": None,
        }

        # Construct URL if needed
        if not url and href:
            url = urljoin(BASE_URL, str(href))
            result["url"] = url

        if not url:
            print(f"[{idx}] No URL or href found, skipping")
            return result

        try:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                }
            )

            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"Status {resp.status_code}, skipping {url}")
                return result

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract audio source URL
            src = None
            tag = soup.select_one(SEL_AUDIO_PRIMARY)
            if tag and tag.has_attr("src"):
                src = tag["src"]
            else:
                # Fallback: look for data-source attribute
                ds_tag = soup.select_one("#ag1 [data-source], [data-source]")
                if ds_tag and ds_tag.has_attr("data-source"):
                    data_src = ds_tag["data-source"]
                    m = re.search(
                        r"(?:mp3|wav)/[A-Za-z0-9._\-\s()%]+(?:\.mp3|\.wav)",
                        data_src,
                        re.IGNORECASE,
                    )
                    if m:
                        src = m.group(0)

            if not src:
                for s in FALLBACK_AUDIO_SELECTORS:
                    t = soup.select_one(s)
                    if t and t.has_attr("src"):
                        src = t["src"]
                        break

            # Normalize source URL to full URL
            audio_url_full = None
            audio_file_name = None
            if src:
                src = src.strip()
                if src.startswith("http://") or src.startswith("https://"):
                    audio_url_full = src
                else:
                    audio_url_full = urljoin(url, src)
                parsed = urlparse(audio_url_full)
                audio_file_name = unquote(os.path.basename(parsed.path))
            else:
                # Last resort: find any source tag with mp3/wav
                found = None
                for s in soup.find_all("source"):
                    ssrc = s.get("src") or s.get("data-src") or ""
                    if re.search(r"\.(mp3|wav)$", ssrc, re.IGNORECASE):
                        found = ssrc
                        break
                if found:
                    audio_url_full = urljoin(url, found)
                    audio_file_name = unquote(os.path.basename(urlparse(audio_url_full).path))

            # Extract audio length
            audio_length = None
            time_tag = soup.select_one(SEL_TIME_PRIMARY)
            if time_tag:
                audio_length = time_tag.get_text(strip=True)
            else:
                for s in FALLBACK_TIME_SELECTORS:
                    t = soup.select_one(s)
                    if t and t.get_text(strip=True):
                        audio_length = t.get_text(strip=True)
                        break

            if not audio_length:
                t = soup.find("div", class_="total-time")
                if t:
                    audio_length = t.get_text(strip=True)

            result.update({
                "audio_length": audio_length,
                "audio_file": audio_file_name,
                "audio_url": audio_url_full,
            })

        except Exception as e:
            print(f"Error on {url}: {e}")

        time.sleep(0.05)
        return result

    def fetch_audio_urls_from_details(self):
        self.logger.info("Fetch audio URL from details...")

        # Prepare arguments: (index, url, href) tuples
        fetch_args = []
        for idx, row in self.sound_details.iterrows():
            url = row.get("url") if pd.notna(row.get("url")) else None
            href = row.get("href") if pd.notna(row.get("href")) else None
            fetch_args.append((idx, url, href))

        # Use multiprocessing pool with 8 workers
        with Pool(processes=8) as pool:
            results = pool.map(SoundBibleScraper._fetch_single_audio_url, fetch_args)

        # Update DataFrame with results
        for result in results:
            idx = result["index"]
            self.sound_details.at[idx, "audio_length"] = result["audio_length"]
            self.sound_details.at[idx, "audio_file"] = result["audio_file"]
            self.sound_details.at[idx, "audio_url"] = result["audio_url"]

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
        # Create list of (url, output_path) tuples for download
        download_args = list(
            zip(
                self.sound_details["audio_url_wav"],
                self.sound_details["sound_location"],
            )
        )
        self.logger.info("Download audio files from URL...")
        with Pool(processes=8) as pool:
            results = pool.starmap(
                SoundBibleScraper._download_sound_effect, download_args
            )
        n = sum([int(i) for i in results])
        self.logger.info(
            f"Downloaded {n} sounds out of {len(results)} ({n / len(results)}%)"
        )

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
