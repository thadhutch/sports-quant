import os
import time
from selenium import webdriver  # Python Dynamic web scraping library
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

def login_to_pff():
    signinurl = "https://auth.pff.com"
    email = os.environ.get("PFF_EMAIL")
    pw = os.environ.get("PFF_PASSWORD")
    email_input = '//*[@id="login-form_email"]'
    pw_input = '//*[@id="login-form_password"]'
    login_submit = '//*[@id="sign-in"]'

    driver = webdriver.Chrome()
    driver.get(signinurl)

    driver.find_element("xpath", email_input).send_keys(email)
    driver.find_element("xpath", pw_input).send_keys(pw)

    time.sleep(3)

    driver.find_element("xpath", login_submit).click()

    return driver

def navigate_and_sign_in(driver, url):
    driver.get(url)
    
    # Wait for the "Sign In" button to appear and click it
    try:
        sign_in_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(@class, "g-btn--green")]'))
        )
        sign_in_button.click()
        time.sleep(3)  # Allow time for the sign-in process to complete if necessary
    except:
        print("Sign In button not found or not needed for this page.")