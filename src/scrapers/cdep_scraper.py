import re
import time
import json
import logging
import ssl
from pathlib import Path
from typing import Any, Optional

import requests
import urllib3
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm  # type: ignore
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry

from src.utils.types import MeetingSession, DialogueTurn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()],
)


class LegacySSLAdapter(HTTPAdapter):
    """
    Custom Adapter to force OpenSSL to accept legacy ciphers (SECLEVEL=1).
    This fixes [SSL: SSLV3_ALERT_HANDSHAKE_FAILURE] often encountered with older government websites.
    """

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = False,
        **pool_kwargs: Any,
    ) -> None:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            ctx.set_ciphers("DEFAULT@SECLEVEL=1")
        except (ValueError, ssl.SSLError):
            ctx.set_ciphers("HIGH:!DH:!aNULL")

        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_context=ctx,
            **pool_kwargs,
        )


class CDEPScraper:
    """
    Scraper for the Romanian Chamber of Deputies (CDEP) meeting transcripts.
    Fetches transcripts, metadata, and summary points from 'steno2015.sumar'.
    """

    BASE_URL: str = "https://www.cdep.ro/pls/steno/steno2015.sumar"

    def __init__(self, output_dir: str = "data/processed") -> None:
        """
        Initializes the scraper with a custom SSL session.

        Args:
            output_dir (str): Directory to save scraped JSON files.
        """

        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session: requests.Session = requests.Session()

        retries = Retry(
            total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504]
        )

        adapter = LegacySSLAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Research Project; Educational)"}
        )
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _get_soup(
        self, url: str, params: Optional[dict[str, Any]] = None
    ) -> Optional[BeautifulSoup]:
        """
        Fetches a URL and returns a BeautifulSoup object.

        Args:
            url (str): The URL to fetch.
            params (dict, optional): Query parameters.

        Returns:
            Optional[BeautifulSoup]: Parsed HTML or None if request failed/no record found.
        """

        try:
            response = self.session.get(url, params=params, timeout=15, verify=False)

            if "Nu exista inregistrare" in response.text:
                return None

            response.raise_for_status()
            return BeautifulSoup(response.content, "lxml")
        except requests.RequestException as e:
            logging.error(f"Failed to fetch {url}: {e}")
            return None

    def _extract_dialogue(self, transcript_url: str) -> tuple[str, list[DialogueTurn]]:
        """
        Extracts the full text and structured dialogue from a transcript page.

        Args:
            transcript_url (str): URL of the specific session transcript.

        Returns:
            tuple[str, list[DialogueTurn]]: A tuple containing the full text and a list of dialogue turns.
        """

        soup = self._get_soup(transcript_url)
        if not soup:
            return "", []

        full_text_parts: list[str] = []
        dialogue_turns: list[DialogueTurn] = []

        paragraphs = soup.find_all("p")
        if len(paragraphs) < 5:
            paragraphs = soup.find_all("td", class_="text")

        current_speaker: str = "UNKNOWN"
        current_text: list[str] = []

        for p in paragraphs:
            raw_text: str = p.get_text(strip=True)
            if not raw_text:
                continue

            if raw_text.startswith("(") and raw_text.endswith(")"):
                continue

            full_text_parts.append(raw_text)

            if ":" in raw_text[:60] and raw_text[0].isupper():
                potential_speaker, content = raw_text.split(":", 1)

                if len(potential_speaker) < 60:
                    if current_text:
                        dialogue_turns.append(
                            DialogueTurn(
                                speaker=current_speaker, text=" ".join(current_text)
                            )
                        )
                    current_speaker = potential_speaker.strip()
                    current_text = [content.strip()]
                else:
                    current_text.append(raw_text)
            else:
                current_text.append(raw_text)

        if current_text:
            dialogue_turns.append(
                DialogueTurn(speaker=current_speaker, text=" ".join(current_text))
            )

        return " ".join(full_text_parts), dialogue_turns

    def scrape_session(self, session_id: int) -> Optional[MeetingSession]:
        """
        Scrapes all data for a single meeting session.

        Args:
            session_id (int): The unique ID of the session (from CDEP database).

        Returns:
            Optional[MeetingSession]: Structured session data or None if failed.
        """

        params = {"ids": session_id, "idl": 1}
        soup = self._get_soup(self.BASE_URL, params=params)

        if not soup:
            return None

        date_tag = soup.find("td", class_="headline")
        date_str: str = date_tag.get_text(strip=True) if date_tag else "Unknown Date"

        summary_points: list[str] = []
        content_table = soup.find("div", id="olddiv") or soup.find("td", class_="text")

        if isinstance(content_table, Tag):
            summary_points = [
                row.get_text(strip=True)
                for row in content_table.find_all("tr")
                if len(row.get_text(strip=True)) > 20
            ]

        link_tag = soup.find("a", string=re.compile(r"Stenograma|Dezbateri"))

        if not link_tag or not isinstance(link_tag, Tag):
            return None

        href = link_tag.get("href")
        if not isinstance(href, str):
            return None

        transcript_url = f"https://www.cdep.ro{href}"
        full_text, dialogue = self._extract_dialogue(transcript_url)

        if not full_text:
            return None

        return MeetingSession(
            session_id=session_id,
            date=date_str,
            summary_points=summary_points,
            full_transcript=full_text,
            dialogue=dialogue,
        )

    def run_batch(self, start_id: int, end_id: int) -> None:
        results: list[dict[str, Any]] = []
        output_file = self.output_dir / f"cdep_{start_id}_{end_id}.json"

        logging.info(f"Starting scrape: IDs {start_id} to {end_id}")

        for sid in tqdm(range(start_id, end_id), desc="Scraping Sessions"):
            session_data = self.scrape_session(sid)
            if session_data:
                results.append(session_data.model_dump())
            time.sleep(0.2)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logging.info(f"Completed. Saved {len(results)} sessions to {output_file}")


if __name__ == "__main__":
    from src.config import START_ID, END_ID
    scraper = CDEPScraper()
    scraper.run_batch(start_id=START_ID, end_id=END_ID)

