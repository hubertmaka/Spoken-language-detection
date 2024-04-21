import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC

language = input("Enter language: ")
my_email = 'does.not.exist.email@gmail.com'
download_folder = r'C:\Users\Admin\Desktop\Python\Projekt Uczenie Maszynowe\Spoken-language-detection'
chrome_options = Options()

chrome_options.add_experimental_option('prefs', {
    "download.default_directory": download_folder,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

service = Service(executable_path='chromedriver.exe')
driver = webdriver.Chrome(service=service)

driver.get('https://commonvoice.mozilla.org/pl/datasets')

WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.NAME, 'bundleLocale'))
)

# option_element = driver.find_element(By.NAME, 'bundleLocale')
# option_element.click()

language_element = driver.find_element(By.XPATH, f'//option[text()="{language}"]')
language_element.click()

# inputing email
wait = WebDriverWait(driver, 10)
email_element = wait.until(EC.presence_of_element_located((By.XPATH, '//div[@class="dataset-download-prompt"]//input')))
email_element = driver.find_element(By.XPATH, '//div[@class="dataset-download-prompt"]//input')
email_element.send_keys(my_email)

# clicking checkbox1
checkbox1 = driver.find_element(By.XPATH, '//input[@name="confirmSize"]')
checkbox1.click()

# clicking checkbox2
checkbox2 = driver.find_element(By.XPATH, '//input[@name="confirmNoIdentify"]')
checkbox2.click()

download_button = driver.find_element(By.XPATH, '//a[@class="button rounded download-language"]')
download_button.click()


def wait_for_file_in_folder(folder_path, timeout=86_400):
    end_time = time.time() + timeout
    while True:
        if any(filename.endswith('.tar.gz') for filename in os.listdir(folder_path)):
            return True
        if time.time() > end_time:
            raise TimeoutError("Download timed out.")
        time.sleep(300)


wait_for_file_in_folder(download_folder)


driver.quit()