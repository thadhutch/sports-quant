import os
import time
from selenium import webdriver  # Python Dynamic web scraping library


url = 'https://premium.pff.com/nfl/teams/2022/REGPO/tampa-bay-buccaneers/schedule'
signinurl = 'https://auth.pff.com'
email = os.environ.get('PFF_EMAIL')
pw = os.environ.get('PFF_PASSWORD')
email_input = '//*[@id="login-form_email"]'
pw_input = '//*[@id="login-form_password"]'
login_submit = '//*[@id="sign-in"]'
continuetostats = '/html/body/div/div/div/div/div/div/ul/li[2]/a'

driver = webdriver.Chrome()
driver.get(signinurl)


driver.find_element("xpath", email_input).send_keys(email)
driver.find_element("xpath", pw_input).send_keys(pw)

time.sleep(3)

driver.find_element("xpath", login_submit).click()

