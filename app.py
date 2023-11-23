import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import yfinance as yf
import datetime

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

ticker = st.sidebar.text_input('Stock Ticker', 'AAPL')
if ticker:
    data_max = get_max_stock_data(ticker)
    current_year = datetime.datetime.now().year
    start_of_year = datetime.datetime(current_year, 1, 1)
    start_date = st.sidebar.date_input('Start Date', start_of_year)
    end_date = st.sidebar.date_input('End Date', datetime.datetime.now())

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
            text="Max High",
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
            text="Min Low",
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

        with st.expander("See detailed data"):
            st.dataframe(data_filtered)

    else:
        st.error('No data found for the given ticker. Please try a different one.')
