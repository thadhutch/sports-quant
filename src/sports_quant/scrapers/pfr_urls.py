"""Gather NFL boxscore URLs from Pro Football Reference."""

import logging
import random
import time

import requests
from bs4 import BeautifulSoup

from sports_quant import _config as config
from sports_quant.scrapers.proxies import load_proxies_from_csv

logger = logging.getLogger(__name__)

# Base URL structure
base_url = "https://www.pro-football-reference.com/years/{}/week_{}.htm"


def get_random_proxy(proxies_list: list) -> dict:
    """Return a random proxy from the list."""
    return random.choice(proxies_list)


def get_boxscores(year: int, week: int, proxies_list: list) -> list:
    """Extract boxscore URLs for a given year and week."""
    url = base_url.format(year, week)
    proxy = get_random_proxy(proxies_list)

    try:
        # Send request using a proxy
        response = requests.get(url, proxies=proxy, timeout=5)

        # Check if the page was successfully fetched
        if response.status_code != 200:
            logger.warning("Failed to fetch %s with proxy %s", url, proxy['http'])
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        boxscores = []

        # Find all td elements with class 'right gamelink'
        for gamelink in soup.find_all('td', class_='right gamelink'):
            a_tag = gamelink.find('a')
            if a_tag and 'href' in a_tag.attrs:
                boxscore_url = "https://www.pro-football-reference.com" + a_tag['href']
                boxscores.append(boxscore_url)

        return boxscores

    except Exception as e:
        logger.error("Error fetching %s with proxy %s: %s", url, proxy['http'], e)
        return []


def collect_boxscore_urls():
    proxies_list = load_proxies_from_csv(str(config.PROXY_FILE), format="requests")

    # Main loop to gather boxscores
    all_boxscores = []

    for year in range(config.START_YEAR, config.END_YEAR + 1):
        # Determine the range of weeks to scrape based on the year
        max_weeks = config.MAX_WEEK if year == config.END_YEAR else 17
        for week in range(1, max_weeks + 1):
            logger.info("Scraping year %d, week %d...", year, week)
            logger.info("Total boxscores scraped so far: %d", len(all_boxscores))
            boxscores = get_boxscores(year, week, proxies_list)
            logger.info("Found %d boxscores", len(boxscores))
            all_boxscores.extend(boxscores)
            time.sleep(1)  # Add a small delay to avoid overwhelming the server

    # Save the results to a file
    with open(config.PFR_BOXSCORES_FILE, 'w') as f:
        for boxscore in all_boxscores:
            f.write(boxscore + '\n')

    logger.info("Scraped %d boxscores.", len(all_boxscores))


if __name__ == "__main__":
    collect_boxscore_urls()
