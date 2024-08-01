import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time

class web_scraping:

    def get_tickers(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=chrome_options)

        url = "https://swaggystocks.com/dashboard/wallstreetbets/ticker-sentiment"

        driver.get(url)

        print("Getting stocks tickers")
        # wait for the dropdown
        wait = WebDriverWait(driver, 10)
        dropdown = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "styles_input__5Rrin")))

        actions = ActionChains(driver)
        actions.move_to_element(dropdown).click().perform()

        time.sleep(1)

        menu_option = wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div/div/div[2]/div[4]/div/div/div/div/div[1]/div/div[1]/div[2]")))
        actions.move_to_element(menu_option).click().perform()

        time.sleep(10)

        current_text = dropdown.text
        print(current_text)

        wait.until(EC.text_to_be_present_in_element((By.CLASS_NAME, "styles_input__5Rrin"), "24 HOURS"))

        def mentions_above_10():
            cards = driver.find_elements(By.CLASS_NAME, "styles_card__4HWKI")
            for card in cards:
                try:
                    mentions_text = card.find_element(By.CLASS_NAME, "styles_mentions__YtuyJ").text
                    mentions = int(mentions_text.split()[0])
                    if mentions <= 10:
                        return False
                except Exception as e:
                    print(f"Error finding mentions element: {e}")
                    return False
            return True

        # click "LOAD MORE" as long as mentions are above 10
        while mentions_above_10():
            try:
                load_more_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'styles_container__3QuxD')]")))
                actions.move_to_element(load_more_button).click().perform()
                print("Clicked 'LOAD MORE' button")
                time.sleep(3)  # Wait for the new content to load
            except Exception as e:
                print("No more 'LOAD MORE' button found or an error occurred:", str(e))
                break


        data = []
        cards = driver.find_elements(By.CLASS_NAME, "styles_card__4HWKI")
        for card in cards:
            ticker = card.find_element(By.CLASS_NAME, "styles_name__fT9wO").text

            data.append({
                "Ticker": ticker,
            })
        driver.quit()
        return data


    def stock_data(self, ticker, period):
        url = f"https://api.beta.swaggystocks.com/memeStocks/sentiment-historical?ticker={ticker}&time={period}"

        response = requests.get(url)

        response_data = response.json()
        data = response_data.get('data', [])

        # calculate the negative sentiment
        extracted_data = []
        for item in data:
            date = item['date']
            social_volume = item['social_volume']
            sentiment_str = item['sentiment']

            if sentiment_str is None:
                sentiment = 0.0
            else:
                sentiment = float(sentiment_str)

            negative_sentiment = 1 - sentiment
            extracted_data.append([date, social_volume, sentiment, negative_sentiment])

        df = pd.DataFrame(extracted_data, columns=['Date', 'Comment volume', 'Positive sentiment', 'Negative sentiment'])

        return df
