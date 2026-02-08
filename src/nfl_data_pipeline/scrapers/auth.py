"""PFF authentication helpers using Selenium."""

import json
import logging
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from nfl_data_pipeline import _config as config

logger = logging.getLogger(__name__)

COOKIE_FILE = config.DATA_DIR / "pff_cookies.json"


def _save_cookies(driver: webdriver.Chrome) -> None:
    """Save browser cookies to disk."""
    COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COOKIE_FILE, "w") as f:
        json.dump(driver.get_cookies(), f)
    logger.info("Cookies saved to %s", COOKIE_FILE)


def _load_cookies(driver: webdriver.Chrome) -> bool:
    """Load cookies from disk into the driver. Returns True if cookies were loaded."""
    if not COOKIE_FILE.exists():
        return False
    try:
        with open(COOKIE_FILE) as f:
            cookies = json.load(f)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Corrupt cookie file, deleting")
        COOKIE_FILE.unlink()
        return False
    driver.get("https://www.pff.com")
    for cookie in cookies:
        cookie.pop("sameSite", None)
        cookie.pop("storeId", None)
        try:
            driver.add_cookie(cookie)
        except Exception:
            pass
    logger.info("Cookies loaded from %s", COOKIE_FILE)
    return True


def login_to_pff() -> webdriver.Chrome:
    """Log into PFF and return an authenticated Chrome driver.

    On first run, opens Chrome and waits for you to log in manually.
    Subsequent runs reuse saved cookies to skip login.
    """
    driver = webdriver.Chrome()

    if _load_cookies(driver):
        driver.get("https://premium.pff.com")
        time.sleep(3)
        if "auth.pff.com" not in driver.current_url:
            logger.info("Logged in via saved cookies")
            return driver
        logger.info("Saved cookies expired, manual login required")

    driver.get("https://auth.pff.com")

    print("\n" + "=" * 60)
    print("Please log into PFF in the Chrome window.")
    print("After you're logged in, press Enter here to continue...")
    print("=" * 60 + "\n")
    input()

    _save_cookies(driver)
    return driver


def navigate_and_sign_in(driver: webdriver.Chrome, url: str) -> None:
    """Navigate to a PFF page and click the sign-in button if prompted."""
    driver.get(url)

    try:
        sign_in_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(@class, "g-btn--green")]'))
        )
        sign_in_button.click()
        time.sleep(3)
    except Exception as e:
        logger.info("Sign In button not found or not needed for this page: %s", e)
