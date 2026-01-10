# Scrapers Module

This module handles the scraping of meeting data from the Romanian Chamber of Deputies (CDEP) website.

## `cdep_scraper.py`

The `CDEPScraper` class orchestrates the data retrieval.

### Key Components

-   **LegacySSLAdapter**: A custom HTTP adapter to force `SECLEVEL=1` ciphers. This is necessary because the CDEP website uses outdated SSL configurations that modern Python/OpenSSL environments reject by default.
-   **Soup Parsing**: Uses `BeautifulSoup` to parse HTML.
-   **Heuristic Extraction**:
    -   Finds the transcript link (looking for "Stenograma" or "Dezbateri").
    -   Extracts dialogue turns by identifying lines starting with Speaker Name (e.g. "Domnul Ion Popescu:").
    -   Filters out procedural text (e.g., "(Aplauze)").
    -   Extracts summary points from the HTML table structure.

### Usage
Run the scraper directly or via the main CLI to fetch a range of session IDs.
The data is saved as a JSON file containing `MeetingSession` objects.
