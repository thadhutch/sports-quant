from selenium import webdriver  # Python Dynamic web scraping library
import time
import pandas as pd

url = 'https://premium.pff.com/nfl/teams/2022/REGPO/tampa-bay-buccaneers/schedule'
signinurl = 'https://auth.pff.com'
email = 'your email here!'
pw = 'your pw here!'
email_input = '//*[@id="login-form_email"]'
pw_input = '//*[@id="login-form_password"]'
login_submit = '//*[@id="sign-in"]'
continuetostats = '/html/body/div/div/div/div/div/div/ul/li[2]/a'


driver = webdriver.Chrome()
driver.get(signinurl)


driver.find_element("xpath", email_input).send_keys(email)
driver.find_element("xpath", pw_input).send_keys(pw)
driver.find_element("xpath", login_submit).click()

# Sign In
driver.find_element("xpath", '//*[@id="react-root"]/div/header/div[3]/button').click()