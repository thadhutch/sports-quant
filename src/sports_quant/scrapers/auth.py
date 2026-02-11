"""PFF authentication helpers using Selenium."""

import logging
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


def login_to_pff() -> webdriver.Chrome:
    """Log into PFF and return an authenticated Chrome driver.

    Opens Chrome and waits for the user to log in manually.
    """
    driver = webdriver.Chrome()
    driver.get("https://auth.pff.com")

    print("\n" + "=" * 60)
    print("Please log into PFF in the Chrome window.")
    print("After you're logged in, press Enter here to continue...")
    print("=" * 60 + "\n")
    input()

    logger.info("Manual login complete")
    return driver


def navigate_and_sign_in(driver: webdriver.Chrome, url: str) -> None:
    """Navigate to a PFF page and click the sign-in button if prompted."""
    driver.get(url)

    try:
        sign_in_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(@class, "g-btn--green")]'))
        )
        sign_in_button.click()
        logger.debug("Clicked sign-in button")
        time.sleep(3)
    except Exception:
        logger.debug("Sign In button not found, likely already authenticated")
