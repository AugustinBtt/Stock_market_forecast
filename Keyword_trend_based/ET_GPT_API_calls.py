import ast
import time
from datetime import datetime, timezone
import pandas as pd
import requests
import openai
from openai import OpenAI

ET_API = "YOUR API KEY"
openai_api_key = "YOUR API KEY"
client = OpenAI(api_key=openai_api_key)


class GetTrendyStocks:
    def get_trends(self, categories):
        url = f"https://api.explodingtopics.com/api/v1/topics?type=exploding&brand=true&categories={categories}&sort=absolute_volume&order=desc&offset=0&limit=250&response_timeframe=last_3_months&api_key={ET_API}"
        # EXPLODING TOPICS OVER THE LAST 3 or 6 MONTHS filtered by ABSOLUTE VOLUME
        response = requests.get(url)
        data = response.json()

        potential_topics = {}
        results = data.get("result", [])

        for item in results:
            topic_key = item.get("keyword")
            if topic_key:
                potential_topics[topic_key] = {
                    "description": item.get("description"),
                    "absolute_volume": item.get("absolute_volume")
                }

        filtered_topics = {key: value for key, value in potential_topics.items() if
                           value["absolute_volume"] is not None and value["absolute_volume"] >= 100000}
        return filtered_topics


    def gpt_get_company_info(self, product_name, description):
        messages = [
            {"role": "system", "content": "You are a knowledgeable assistant in financial markets."},
            {"role": "user", "content": (
                f"I have a brand named '{product_name}'.\n"
                f"Here is a description of the business/product: {description}.\n"
                "Please provide the following information:\n"
                "1. The name of the company making this brand.\n"
                "2. Whether this company is publicly traded.\n"
                "3. If publicly traded, the ticker symbol on the exchange with the highest trading volume."
                "Please provide the following information in the form of a Python dictionary:\n"
                "{\n"
                "'brand': ,\n"
                "'company_name': ,\n"
                "'publicly_traded': yes/no,\n"
                "'ticker': \n"
                "}"
                "If there is no ticker write 'none'."
            )}
        ]
        retries = 5
        for i in range(retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=200
                )
                return response.choices[0].message.content.strip()
            except openai.RateLimitError as e:
                if i < retries - 1:
                    sleep_time = 2 ** i  # exponential backoff
                    print(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print("Rate limit exceeded. Maximum retries reached.")
                    raise e


    def gpt_get_companies(self, topics_trending_dict):
        publicly_traded_companies = {}
        for company_name, details in topics_trending_dict.items():
            description = details['description']
            company_info_text = self.gpt_get_company_info(company_name, description)
            print(f"Company Info for {company_name}:\n{company_info_text}\n")

            company_info_text = company_info_text.replace("'", '"')

            # ast.literal_eval to convert the string response to a dictionary
            try:
                company_info = ast.literal_eval(company_info_text)
                if company_info.get("publicly_traded", "") in ["yes", "public"]:
                    publicly_traded_companies[company_name] = {
                        "company_name": company_info.get("company_name"),
                        "ticker": company_info.get("ticker")
                    }
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing company info for {company_name}: {company_info_text}")
                print(f"Exception: {e}")

        return publicly_traded_companies


class Volume:
    def __init__(self, arg):
        self.arg = arg
        url = f"https://api.explodingtopics.com/api/v1/topic?keyword={self.arg}&response_timeframe=last_15_years&api_key={ET_API}"
        response = requests.get(url)
        data = response.json()
        self.search_data = data.get('result', {}).get('search_history', {})

    def get_volume(self):
        last_15_years_data = self.search_data.get('last_15_years', [])
        volume_dic = {}
        for entry in last_15_years_data:
            time_key = entry.get('time')
            if time_key:
                formatted_date = datetime.fromtimestamp(int(time_key), tz=timezone.utc).strftime('%Y-%m-%d')
                volume_dic[formatted_date] = {
                    "volume": entry.get('value')
                }
        volume_df = pd.DataFrame.from_dict(volume_dic, orient='index').reset_index()
        volume_df.columns = ['date', 'volume']
        return volume_df


    # THE API GIVES US ACCESS TO THE SEARCH VOLUME FORECAST FOR EVERY TOPIC
    # def get_forecast(self):
    #     volume_forecast = self.search_data.get('next_12_months_forecast_monthly', [])
    #     forecast_dic = {}
    #     for entry in volume_forecast:
    #         time_key = entry.get('time')
    #         if time_key:
    #             formatted_date = datetime.fromtimestamp(int(time_key), tz=timezone.utc).strftime('%Y-%m-%d')
    #             forecast_dic[formatted_date] = {
    #                 "volume": entry.get('value')
    #             }
    #     forecast_df = pd.DataFrame.from_dict(forecast_dic, orient='index').reset_index()
    #     forecast_df.columns = ['date', 'volume']
    #     return forecast_df
