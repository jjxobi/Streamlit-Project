import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import math
import plotly.express as px
import plotly.graph_objs as go
import yfinance as yf
import datetime
import numpy as np
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from newsapi import NewsApiClient
from textblob import TextBlob

# Initialize a text generation pipeline
generator = pipeline('text-generation', model='gpt2')

def generate_insights_text(data):
    # Analyze the data to generate a summary text
    trend = "upward" if data.iloc[-1]['Close'] > data.iloc[0]['Close'] else "downward"
    max_close = round(data['Close'].max(), 2)
    min_close = round(data['Close'].min(), 2)

    # Additional insights
    average_volume = round(data['Volume'].mean(), 2)
    recent_performance = "improved" if data.iloc[-1]['Close'] > data.iloc[-30]['Close'] else "declined"

    insights_text = (
        f"The stock has shown a {trend} trend over the selected period, with the highest closing price at ${max_close} "
        f"and the lowest at ${min_close}. The average trading volume in this period was {average_volume}. "
        f"In the most recent month, the stock's performance has {recent_performance}. "
        f"This trend analysis indicates potential market movements and investor sentiment."
    )
    return insights_text

def generate_summary(text):
    # Ensure the text is not too short
    if len(text.split()) < 10:  # Arbitrary minimum length
        return "Input text is too short for summary generation."

    # Truncate the text if it's too long
    max_tokens = 1024  # Max tokens for GPT-2
    truncated_text = " ".join(text.split()[:max_tokens])

    # Generate summary
    try:
        # Adjust max_length relative to the input length
        summary_length = min(len(truncated_text.split()) * 2, 50)
        summary = generator(truncated_text, max_length=summary_length, num_return_sequences=1)[0]
        return summary['generated_text']
    except Exception as e:
        return f"An error occurred in text generation: {e}"


# Page Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page:", ["Stock Dashboard", "Portfolio Tracker"])

if page == "Stock Dashboard":

    newsapi = NewsApiClient(api_key='51545828af3c49b49443fc10d317a5a6')

    st.title('Real-Time Stock Market Dashboard')
    st.sidebar.header('User Input')

    @st.cache_data
    def get_max_stock_data(ticker):
        try:
            data = yf.download(ticker)
            if data.empty:
                st.error('No data found for the given ticker. Please try a different one.')
                return pd.DataFrame()
        except Exception as e:
            st.error(f'An error occurred while fetching data: {e}')
            return pd.DataFrame()
        return data

    # Function to fetch news
    def get_news(ticker):
        all_articles = newsapi.get_everything(q=ticker,
                                            language='en',
                                            sort_by='relevancy')
        return all_articles['articles']

    # Function to perform sentiment analysis
    def analyze_sentiment(headline):
        analysis = TextBlob(headline)
        return analysis.sentiment.polarity

    def get_sentiment_emoji(score):
        if score > 0:
            return '😊', 'Positive', 'green'  # Smiling face for positive sentiment
        elif score < 0:
            return '😠', 'Negative', 'red'  # Angry face for negative sentiment
        else:
            return '😐', 'Neutral', 'yellow'  # Straight face for neutral sentiment

    ticker = st.sidebar.text_input('Stock Ticker', 'AAPL')
    if ticker:
        stock_info = yf.Ticker(ticker).info
        stock_name = stock_info.get('longName', ticker)
        st.header(f'{stock_name} ({ticker})')

        data_max = get_max_stock_data(ticker)
        current_year = datetime.datetime.now().year
        start_of_year = datetime.datetime(current_year, 1, 1)
        start_date = st.sidebar.date_input('Start Date', start_of_year)
        end_date = st.sidebar.date_input('End Date', datetime.datetime.now())

        st.sidebar.subheader('Latest News and Sentiment')
        news_articles = get_news(ticker)

        for article in news_articles[:5]:  # Display the top 5 news articles
            sentiment_score = analyze_sentiment(article['title'])
            emoji, sentiment, color = get_sentiment_emoji(sentiment_score)

            # News article container
            with st.container():
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    # Making title clickable but not look like a hyperlink
                    st.markdown(f'<a href="{article["url"]}" target="_blank" style="text-decoration: none; color: white;">{article["title"]}</a>', unsafe_allow_html=True)
                    if article['urlToImage']:
                        try:
                            response = requests.get(article['urlToImage'])
                            response.raise_for_status()  # Will raise an HTTPError for bad status
                            img = Image.open(BytesIO(response.content))
                            st.image(img, width=100)
                        except Exception:
                            # If there's an error, skip the image but continue with the rest
                            pass
                with col2:
                    # Display sentiment emoji with hover text
                    st.markdown(f'<div style="display: flex; align-items: center; justify-content: center; height: 100%;"><span title="{sentiment}" style="color: {color}; font-size: 20px;">{emoji}</span></div>', unsafe_allow_html=True)
                
                # Adding a border to the article container
                st.sidebar.markdown("---")

        if not data_max.empty:
            data_filtered = data_max.loc[start_date:end_date]
            open_price = data_filtered.iloc[0]['Open']
            max_high = data_filtered['High'].max()
            min_low = data_filtered['Low'].min()
            close_price = data_filtered.iloc[-1]['Close']
            adj_close = data_filtered.iloc[-1]['Adj Close']

            if close_price > data_filtered.iloc[-2]['Close']:
                trend = '↗️'
                trend_color = "green"
            elif close_price < data_filtered.iloc[-2]['Close']:
                trend = '↘️'
                trend_color = "red"
            else:
                trend = '➡️'
                trend_color = "grey"

            summary_data = pd.DataFrame({
                'Open': [open_price],
                'High': [max_high],
                'Low': [min_low],
                'Close': [close_price],
                'Adj Close': [adj_close],
                'Closing Trend': [trend]
            }).reset_index(drop=True)

            sparkline_data = data_max['Close'].iloc[-14:]
            sparkline_fig = px.line(sparkline_data, title="14 Day Trend")
            sparkline_fig.update_layout(
                title={
                    'text': 'Current 14 Day Trend',
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 10, 'family': "Arial, Helvetica, sans-serif"}
                },
                showlegend=False,
                xaxis=dict(title='', showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(title='', showticklabels=False, showgrid=False, zeroline=False),
                margin=dict(l=0, r=0, t=0, b=0),
                height=100  # Adjust the height to match the table
            )

            col1, col2 = st.columns([3, 1])
            with col1:
                st.table(summary_data)
            with col2:
                st.plotly_chart(sparkline_fig, use_container_width=True)

            # Plot the closing price as a line chart with high and low annotations
            fig = px.line(data_filtered, x=data_filtered.index, y='Close', title='Closing Price Over Time')

            # Find the positions for the annotations
            high_point = data_filtered['High'].idxmax()
            low_point = data_filtered['Low'].idxmin()
            max_high = data_filtered['High'].max()
            min_low = data_filtered['Low'].min()

            # Add annotations with arrows pointing to the max and min
            fig.add_annotation(
                x=high_point,
                y=max_high,
                text="High",
                showarrow=True,
                arrowhead=1,
                arrowsize=2,
                arrowwidth=2,
                arrowcolor="#FF0000",
                ax=0,
                ay=-40
            )
            fig.add_annotation(
                x=low_point,
                y=min_low,
                text="Low",
                showarrow=True,
                arrowhead=1,
                arrowsize=2,
                arrowwidth=2,
                arrowcolor="#0000FF",
                ax=0,
                ay=40
            )

            # Ensure the entire chart is displayed
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1M', step='month', stepmode='backward'),
                            dict(count=6, label='6M', step='month', stepmode='backward'),
                            dict(count=1, label='YTD', step='year', stepmode='todate'),
                            dict(count=1, label='1Y', step='year', stepmode='backward'),
                            dict(step='all')
                        ])
                    )
                )
            )

            st.plotly_chart(fig)

            #Predictive analaysis 
            # Fetch additional historical data for LSTM training
            two_years_ago = datetime.datetime.now() - datetime.timedelta(days=730)
            data_for_model = yf.download(ticker, start=two_years_ago.strftime('%Y-%m-%d'))

            # Preprocess data for LSTM
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data_model = scaler.fit_transform(data_for_model['Close'].values.reshape(-1,1))

            # Create training data set
            train_data_len_model = len(scaled_data_model) - 180  # Exclude the last 6 months (3 months for test, 3 months for prediction)
            train_data_model = scaled_data_model[0:train_data_len_model, :]

            # Convert training data to the format expected by LSTM (samples, time steps, features)
            x_train, y_train = [], []
            for i in range(60, len(train_data_model)):
                x_train.append(train_data_model[i-60:i, 0])
                y_train.append(train_data_model[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Build LSTM model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dense(units=25))
            model.add(Dense(units=1))

            # Compile and train the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

            # Prediction Phase
            test_data = scaled_data_model[train_data_len_model - 60:, :]
            x_test_model = []
            for i in range(60, len(test_data)):
                x_test_model.append(test_data[i-60:i, 0])

            x_test_model = np.array(x_test_model)
            x_test_model = np.reshape(x_test_model, (x_test_model.shape[0], x_test_model.shape[1], 1))

            # Predicting prices
            predicted_prices_model = model.predict(x_test_model)
            predicted_prices_model = scaler.inverse_transform(predicted_prices_model)

            # Ensure that the length of valid_model matches the length of predicted_prices_model
            valid_model = data_for_model.tail(len(predicted_prices_model))
            valid_model['Predictions'] = predicted_prices_model

            # Visualization
            fig_model = px.line()
            fig_model.add_scatter(x=valid_model.index, y=valid_model['Close'], mode='lines', name='Actual Price')
            fig_model.add_scatter(x=valid_model.index, y=valid_model['Predictions'], mode='lines', name='Predictions')
            st.plotly_chart(fig_model)

            # Prepare data for predicting future prices
            last_known_data = scaled_data_model[-60:]  # Last 60 known data points
            current_batch = np.array([last_known_data])  # Reshape to fit LSTM input
            future_period = 90

            # Predict future prices
            predicted_future_prices = []
            for i in range(future_period):
                # Generate prediction for the next day
                prediction = model.predict(current_batch)[0]
                
                # Append prediction to the list of future predictions
                predicted_future_prices.append(prediction)  
                
                # Update the current batch to use the latest prediction
                current_batch = np.append(current_batch[:,1:,:], [[prediction]], axis=1)

            # Inverse transform the predicted prices to original scale
            predicted_future_prices = scaler.inverse_transform(predicted_future_prices)

            # Generate future dates starting from the last known date in the actual data
            last_actual_date = data_for_model.index[-1]
            future_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=future_period)

            # Visualization of historical and predicted prices
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=data_for_model.index, y=data_for_model['Close'], mode='lines', name='Historical Price'))
            fig_future.add_trace(go.Scatter(x=future_dates, y=predicted_future_prices.flatten(), mode='lines', name='Future Predictions'))

            fig_future.update_layout(
                title="Historical and Predicted Prices",
                xaxis_title="Date",
                yaxis_title="Price",
                legend_title="Legend",
                xaxis_rangeslider_visible=False
            )

             # Generate insights text based on data analysis
            insights_text = generate_insights_text(data_filtered)
            trend_summary = generate_summary(insights_text)

            # Display the summary
            st.subheader("Detailed Trend Analysis")
            st.write(trend_summary)

            st.plotly_chart(fig_future)

            with st.expander("See detailed data"):
                st.dataframe(data_filtered)

        else:
            st.error('No data found for the given ticker. Please try a different one.')

elif page == "Portfolio Tracker":
    st.title('Stock Portfolio')

    # Initialize session state for holding tickers and allocations for the portfolio
    if 'portfolio_tickers' not in st.session_state:
        st.session_state.portfolio_tickers = []

    # Sidebar inputs for number of stocks and tickers
    st.sidebar.subheader("Portfolio Tracking")
    num_stocks = st.sidebar.number_input("Number of Stocks in Portfolio", min_value=1, max_value=10, value=max(1, len(st.session_state.portfolio_tickers)), step=1, key="num_stocks")

    # Text input for user to enter a ticker
    new_ticker = st.sidebar.text_input("Enter a stock ticker and press Enter:", key="new_ticker_portfolio")
    
    # Button to add the entered ticker to the list
    if st.sidebar.button("Add Ticker", key="add_ticker_portfolio") or new_ticker:
        if new_ticker.upper() not in st.session_state.portfolio_tickers:
            st.session_state.portfolio_tickers.append(new_ticker.upper())
            st.session_state.new_ticker_portfolio = ""  # Clear the input box after adding
        else:
            st.sidebar.error("Ticker already in the portfolio list.")

    # Show the list of tickers with a multiselect allowing users to remove tickers
    selected_tickers = st.sidebar.multiselect(
        "Selected Tickers (remove any you don't want):",
        st.session_state.portfolio_tickers,
        default=st.session_state.portfolio_tickers,
        key="selected_portfolio_tickers"
    )

    # Update the session state with the selected tickers
    st.session_state.portfolio_tickers = selected_tickers[:num_stocks]  # Limit to number of stocks set

    # Display percentage inputs for the selected tickers
    allocations = {ticker: st.sidebar.number_input(f"Allocation for {ticker} (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"allocation_{ticker}") for ticker in st.session_state.portfolio_tickers}

    # Check if allocations sum to 100%
    if sum(allocations.values()) != 100.0:
        st.sidebar.error("Allocations must sum to 100%")

    # Display tickers and their allocations
    st.write("Portfolio Tickers and Allocations:")
    for ticker, allocation in allocations.items():
        st.write(f"{ticker}: {allocation}%")

    # Calculate Portfolio Value Over Time
    if sum(allocations.values()) == 100.0:
        portfolio_value = pd.DataFrame()
        for ticker, allocation in allocations.items():
            stock_data = get_max_stock_data(ticker)
            if stock_data is not None and not stock_data.empty:
                # Normalize the stock prices to start at the same point in time
                normalized_stock = stock_data['Adj Close'] / stock_data['Adj Close'].iloc[0]
                weighted_stock = normalized_stock * (allocation / 100.0)
                portfolio_value[ticker] = weighted_stock

        # Sum across rows to get the portfolio value over time
        portfolio_value['Total'] = portfolio_value.sum(axis=1)

        # Display Portfolio Value Chart
        st.line_chart(portfolio_value['Total'])

