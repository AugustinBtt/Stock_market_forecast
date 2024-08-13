import os
import numpy as np
import requests
import pandas as pd
import sns
from scipy.interpolate import CubicSpline
from datetime import datetime
from ET_GPT_API_calls import GetTrendyStocks, Volume
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Conv1D, Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import Huber


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

file_path = './CPI_data.csv'
cpi_data = pd.read_csv(file_path)
cpi_data['DATE'] = cpi_data['DATE'].astype(str)

start_date = "2018-03-30"
end_date = "2024-08-07"

fmp_api_key = "YOUR API KEY"


def convert_to_datetime(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")


get_stocks = GetTrendyStocks()

# Exploding Topics API calls
industries = ["business", "technology", "health", "finance", "industry", "fashion"]

industry_trends = {}
for industry in industries:
    industry_trends[industry] = get_stocks.get_trends(industry)

companies_data = {}
# Call GPT to identify the ticker of the company making the brand
for industry, trends in industry_trends.items():
    print(f"Getting trends tickers for industry: {industry}")
    industry_companies = get_stocks.gpt_get_companies(trends)
    companies_data.update(industry_companies)

print(f"ALL COMPANIES: {companies_data}")

for key, details in companies_data.items():
    topic_selected = key
    company = details['company_name'].replace(' Corporation', '')
    ticker = details['ticker'].upper()

    print(f"Processed: Topic: {key}| {company}| {ticker}")


    # DATA PREPARATION
    def get_historical_price():
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={fmp_api_key}"

        response = requests.get(url)
        data = response.json()

        historical_price = {}
        historical_data = data.get('historical', [])
        for item in historical_data:
            date = item.get("date").split(' ')[0]
            close_price = item.get("close")
            historical_price[date] = close_price
        return historical_price


    def get_closest_cpi_qtrdate(financial_date_str):
        financial_date = datetime.strptime(financial_date_str, "%Y-%m-%d")
        year = financial_date.year
        month = financial_date.month

        if month in [1, 4, 7, 10]:
            # CPI date is the same as the financial date
            cpi_date_str = financial_date.strftime("%Y%m") + '01'
        else:
            if month in [2, 5, 8, 11]:
                # first of the same quarter
                month -= 1
            else:
                # for months at the end of a quarter, attempt to use the next quarter's first month
                month += 1 if month != 12 else -11
                year += 0 if month != 1 else 1

            cpi_date_str = f"{year}{str(month).zfill(2)}01"

            # if financial date is in December and there is no CPI data for January next year,
            # revert to October of the current year
            if month == 1 and cpi_date_str not in cpi_data['DATE'].values:
                cpi_date_str = f"{year - 1}1001"
        return cpi_date_str


    def adjust_to_inflation(financial_data, cpi_df):
        adjusted_financial_data = {}
        cpi_df['CPALTT01USQ657N'] = cpi_df['CPALTT01USQ657N'].astype(float) / 100

        for date_str, finance in financial_data.items():
            closest_date_str = get_closest_cpi_qtrdate(date_str)

            initial_revenue = int(finance['revenue'])
            initial_income = int(finance['income'])
            cpi_records = cpi_df[(cpi_df['DATE'] >= closest_date_str)]

            final_revenue = initial_revenue
            final_income = initial_income
            for cpi_change in cpi_records['CPALTT01USQ657N']:
                final_revenue *= (1 + cpi_change)
                final_income *= (1 + cpi_change)

            adjusted_financial_data[date_str] = {
                'adjusted_revenue': final_revenue,
                'adjusted_income': final_income
            }
        return adjusted_financial_data


    start_date_dt = convert_to_datetime(start_date)
    end_date_dt = convert_to_datetime(end_date)


    def get_financial_data():
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=quarterly&apikey={fmp_api_key}"

        response = requests.get(url)
        data = response.json()

        nominal_financial_data = {}
        for item in data:
            end_fiscal_date = item.get("date")
            if end_fiscal_date:
                fiscal_date_dt = datetime.strptime(end_fiscal_date, "%Y-%m-%d")
                if start_date_dt <= fiscal_date_dt <= end_date_dt:
                    nominal_financial_data[end_fiscal_date] = {
                        "revenue": item.get("revenue"),
                        "income": item.get("netIncome")
                    }
        return adjust_to_inflation(nominal_financial_data, cpi_data)


    financial_df = pd.DataFrame(get_financial_data()).transpose()
    if financial_df.empty:
        print("Financial dataframe empty, TICKER INVALID")
        continue
    else:
        financial_df.index = pd.to_datetime(financial_df.index)
        financial_df.sort_index(inplace=True)

    # range for interpolation
    date_range = pd.date_range(start=start_date_dt, end=end_date_dt, freq='D')

    cs_revenue = CubicSpline(financial_df.index.astype(np.int64) // 10 ** 9, financial_df['adjusted_revenue'],
                             bc_type='natural')
    cs_income = CubicSpline(financial_df.index.astype(np.int64) // 10 ** 9, financial_df['adjusted_income'],
                            bc_type='natural')

    daily_revenue = cs_revenue(date_range.astype(np.int64) // 10 ** 9)
    daily_income = cs_income(date_range.astype(np.int64) // 10 ** 9)

    daily_financial_df = pd.DataFrame({
        'Daily Revenue': daily_revenue,
        'Daily Income': daily_income
    }, index=date_range)

    # Generate rolling averages for the start gap
    start_gap_dates = pd.date_range(start=start_date_dt, end=financial_df.index.min() - pd.Timedelta(days=1), freq='D')
    rolling_window = 90
    daily_financial_df['Rolling Revenue'] = daily_financial_df['Daily Revenue'].rolling(window=rolling_window,
                                                                                        min_periods=1).mean()
    daily_financial_df['Rolling Income'] = daily_financial_df['Daily Income'].rolling(window=rolling_window,
                                                                                      min_periods=1).mean()

    start_rolling_revenue = daily_financial_df['Rolling Revenue'].iloc[0]
    start_rolling_income = daily_financial_df['Rolling Income'].iloc[0]

    start_rolling_df = pd.DataFrame({
        'Daily Revenue': [start_rolling_revenue] * len(start_gap_dates),
        'Daily Income': [start_rolling_income] * len(start_gap_dates)
    }, index=start_gap_dates)

    # end gap
    price_data = get_historical_price()
    stock_prices_df = pd.DataFrame(list(price_data.values()), index=pd.to_datetime(list(price_data.keys())),
                                   columns=['Close'])

    # Aligning stock prices with financial data date range
    stock_prices_df = stock_prices_df[stock_prices_df.index >= start_date_dt]

    print("Stock prices date range:", stock_prices_df.index.min(), "to", stock_prices_df.index.max())
    print("Financial data date range:", daily_financial_df.index.min(), "to", daily_financial_df.index.max())

    # gap period at the end
    last_financial_date = daily_financial_df.index.max()
    last_stock_date = stock_prices_df.index.max()
    end_gap_dates = pd.date_range(start=last_financial_date + pd.Timedelta(days=1), end=last_stock_date, freq='D')

    # rolling averages for the end gap
    end_rolling_revenue = daily_financial_df['Rolling Revenue'].iloc[-1]
    end_rolling_income = daily_financial_df['Rolling Income'].iloc[-1]

    end_rolling_df = pd.DataFrame({
        'Daily Revenue': [end_rolling_revenue] * len(end_gap_dates),
        'Daily Income': [end_rolling_income] * len(end_gap_dates)
    }, index=end_gap_dates)

    combined_df = pd.concat([start_rolling_df, daily_financial_df[['Daily Revenue', 'Daily Income']], end_rolling_df])
    combined_df.sort_index(inplace=True)

    # Align with stock closure
    final_df = stock_prices_df.join(combined_df, how='left')



    print("ADDING SEARCH VOLUME")
    vol_nd_forecast = Volume(topic_selected)
    volume_df = vol_nd_forecast.get_volume()
    if volume_df.empty:
        print("No data available in the DataFrame")
        continue
    else:
        volume_df['date'] = pd.to_datetime(volume_df['date'])
        dataframe_start_date = final_df.index.min()
        dataframe_end_date = final_df.index.max()

        # Filter the volume_df to start from the same month as the start date
        volume_df = volume_df[volume_df['date'] >= dataframe_start_date.replace(day=1)].reset_index(drop=True)

        daily_date_range = pd.date_range(start=dataframe_start_date, end=dataframe_end_date, freq='D')
        daily_volume_df = pd.DataFrame(daily_date_range, columns=['date'])
        daily_volume_df = daily_volume_df.merge(volume_df, on='date', how='left')

        daily_volume_df['volume'] = daily_volume_df['volume'].interpolate(method='pchip')

        # rolling average for the volume
        rolling_window = 90
        daily_volume_df['Rolling Volume'] = daily_volume_df['volume'].rolling(window=rolling_window,
                                                                              min_periods=1).mean()

        # fill NaN values with rolling averages
        daily_volume_df['volume'] = daily_volume_df['volume'].combine_first(daily_volume_df['Rolling Volume'])

        daily_volume_df.drop(columns=['Rolling Volume'], inplace=True)

        daily_volume_df.set_index('date', inplace=True)
        daily_volume_df['volume'] = daily_volume_df['volume'].ffill().bfill()

        final_df = final_df.merge(daily_volume_df, left_index=True, right_index=True, how='left',
                                  suffixes=('', '_volume'))

    final_df.sort_index(ascending=False, inplace=True)
    print(f"Final df after merging volume data for {ticker}: \n{final_df}")


    # Normalize
    features = final_df.drop('Close', axis=1)
    target = final_df['Close']
    columns_to_normalize = ['Daily Revenue', 'Daily Income', 'volume']

    scaler_features = MinMaxScaler()
    features_scaled = features.copy()
    features_scaled[columns_to_normalize] = scaler_features.fit_transform(features[columns_to_normalize])

    scaler_target = MinMaxScaler()
    target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

    target_scaled_df = pd.DataFrame(target_scaled, columns=['Close'], index=target.index)
    df_scaled = pd.concat([features_scaled, target_scaled_df], axis=1)

    columns = ['Close'] + [col for col in df_scaled.columns if col != 'Close']
    df_scaled = df_scaled[columns]

    print(f"Scaled DataFrame: {df_scaled}")


    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            label = data[i + seq_length][df_scaled.columns.get_loc('Close')]
            sequences.append((seq, label))
        return sequences

    seq_length = 22
    sequences = create_sequences(df_scaled.values, seq_length)


    X = np.array([seq for seq, label in sequences])
    y = np.array([label for seq, label in sequences])

    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

    model.add(LSTM(64, return_sequences=True, activation='tanh', recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128, return_sequences=True, activation='tanh', recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(LSTM(256, return_sequences=True, activation='tanh', recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(LSTM(64, return_sequences=True, activation='tanh', recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(32, return_sequences=False, activation='tanh', recurrent_dropout=0.2))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='linear'))


    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=Huber())
    model.summary()


    # TRAIN MODEL
    history = model.fit(X_train, y_train, batch_size=64, epochs=350, validation_split=0.2)

    train_loss = model.evaluate(X_train, y_train)
    test_loss = model.evaluate(X_test, y_test)

    print("Train Loss:", train_loss)
    print("Test Loss:", test_loss)

    predictions = model.predict(X_test)
    predictions = predictions.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # inverse transform for actual values
    predictions = scaler_target.inverse_transform(predictions)
    y_test = scaler_target.inverse_transform(y_test)

    # get latest sequence
    latest_sequence = df_scaled.tail(seq_length).values
    latest_sequence = latest_sequence.reshape((1, seq_length, latest_sequence.shape[1]))


    future_predictions = []
    num_days_ahead = 20
    for _ in range(num_days_ahead):
        future_prediction = model.predict(latest_sequence)

        print("Future prediction shape:", future_prediction.shape)

        future_prediction_value = future_prediction[0, 0]

        future_prediction_value = scaler_target.inverse_transform([[future_prediction_value]])[0, 0]

        future_predictions.append(future_prediction_value)

        new_sequence = np.append(
            latest_sequence[0, 1:, :],
            np.array([[future_prediction_value] + [0] * (latest_sequence.shape[2] - 1)]),
            axis=0
        )
        new_sequence = new_sequence[:seq_length, :]

        latest_sequence = new_sequence.reshape((1, seq_length, latest_sequence.shape[2]))


    last_date = df_scaled.index[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, num_days_ahead + 1)]

    forecast_df = pd.DataFrame(data={'date': future_dates, 'predicted_close': future_predictions})

    train_loss_round = round(train_loss, 6)
    test_loss_round = round(test_loss, 6)
    folder_name = 'Predictions'
    file_path = os.path.join(folder_name, f'{ticker} chart.png')

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title(
        f'{ticker} Train loss:{train_loss_round} | Test loss:{test_loss_round} | Prediction: ${forecast_df.iloc[4]['predicted_close']}')
    plt.savefig(file_path)
    plt.close()

    print(forecast_df)
