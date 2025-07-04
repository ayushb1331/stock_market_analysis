import base64
from typing import Optional
from io import StringIO
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_components_plotly, plot_plotly
import yfinance as yf

###############################################################################
# Helper – optional background image                                          #
###############################################################################

def _get_base64(bin_file: str) -> str:
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()


def set_background(png_file: str):
    """Set a full-screen background image inside Streamlit."""
    bin_str = _get_base64(png_file)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

###############################################################################
# Load data – upload or download                                              #
###############################################################################

st.title("📈 Simple Stock-Price Prediction App (Prophet)")

# Initialize session state for downloaded data and symbol
# This is crucial so that the downloaded data persists across reruns
# without needing to re-download if the user navigates pages.
if 'data' not in st.session_state:
    st.session_state.data = None
if 'symbol_title' not in st.session_state:
    st.session_state.symbol_title = ""
if 'prophet_df' not in st.session_state:
    st.session_state.prophet_df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'forecast' not in st.session_state:
    st.session_state.forecast = None
if 'data_loaded' not in st.session_state: # New flag for data loading status
    st.session_state.data_loaded = False


source = st.sidebar.selectbox(
    "Load data via…",
    (
        "Download from Yahoo Finance", # Prioritize download
        "Upload CSV",
    ),
)

# ────────────────────────────────────────────────────────────────────────────
# 1. Upload CSV
# ────────────────────────────────────────────────────────────────────────────
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload Yahoo Finance CSV", type="csv", accept_multiple_files=False
    )
    if uploaded is not None:
        try:
            # Added parse_dates to ensure 'Date' column is parsed correctly
            uploaded_data = pd.read_csv(uploaded, parse_dates=['Date'])
            st.session_state.data = uploaded_data
            st.session_state.symbol_title = uploaded.name.split(".")[0].upper()
            st.success(f"✅ Uploaded {len(st.session_state.data)} rows for **{st.session_state.symbol_title}**")
            # Set a flag to trigger reprocessing
            st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.session_state.data = None
            st.session_state.symbol_title = ""
            st.session_state.data_loaded = False
    else:
        st.info("Upload a CSV file or switch to *Download*.")
        st.session_state.data_loaded = False
        # If no file is uploaded, ensure previous data is cleared to prevent
        # using stale data if the user switches back and forth.
        st.session_state.data = None
        st.session_state.symbol_title = ""
        st.session_state.prophet_df = None
        st.session_state.model = None
        st.session_state.forecast = None

# ────────────────────────────────────────────────────────────────────────────
# 2. Download using yfinance (Ticker.history avoids timezone bug)
# ────────────────────────────────────────────────────────────────────────────
if source == "Download from Yahoo Finance":
    ticker = st.sidebar.text_input("Ticker", "AAPL", max_chars=8).upper().strip()
    col1, col2 = st.sidebar.columns(2)
    # Defaulting start date to a reasonable past date
    start_date = col1.date_input("Start date", pd.to_datetime("2023-01-01"))
    # Ensure end date is not in the future for current data
    end_date = col2.date_input("End date", pd.to_datetime("today"))
    fetch = st.sidebar.button("🔽 Fetch data")

    # Only fetch if the button is clicked OR if it's the first run and data isn't loaded
    # and the current ticker selection is the default 'AAPL' and no data is in session_state.
    # This prevents re-downloading every time if user just navigates.
    if fetch or (st.session_state.data is None and st.session_state.symbol_title == "" and ticker == "AAPL" and not st.session_state.data_loaded):
        if start_date >= end_date:
            st.error("❌ *Start* must be before *End*.")
            st.session_state.data_loaded = False
            st.stop()
        with st.spinner(f"Downloading {ticker} data from Yahoo Finance…"):
            try:
                ticker_obj = yf.Ticker(ticker)
                # Adding 1 day to end_date ensures that the end_date itself is included
                downloaded_data = ticker_obj.history(start=start_date, end=end_date + pd.Timedelta(days=1), interval="1d")
            except Exception as e:
                st.error(f"Download failed: {e}. Please check the ticker symbol or your internet connection.")
                st.session_state.data_loaded = False
                st.stop()

            if downloaded_data is None or downloaded_data.empty:
                st.error("No data returned – check symbol, date range, or try again later.")
                st.session_state.data_loaded = False
                st.stop()

            # history returns index as DatetimeIndex; reset to column
            downloaded_data.reset_index(inplace=True)
            # Rename 'Date' column to 'Date' if it's 'Datetime'
            if 'Datetime' in downloaded_data.columns:
                downloaded_data.rename(columns={'Datetime': 'Date'}, inplace=True)

            st.session_state.data = downloaded_data
            st.session_state.symbol_title = ticker
            st.success(f"✅ Downloaded {len(st.session_state.data)} rows for **{st.session_state.symbol_title}**")
            st.session_state.data_loaded = True # Flag to indicate data is ready for processing

# Guard: we need data to proceed
if st.session_state.data is None or st.session_state.data.empty:
    st.info("Please upload a CSV or fetch data from Yahoo Finance to proceed.")
    st.stop()

###############################################################################
# Prepare dataset for Prophet                                                 #
###############################################################################

# Only re-process data if new data is loaded or if it's the first time processing
if st.session_state.data_loaded or st.session_state.prophet_df is None:
    data_to_process = st.session_state.data.copy()

    # Ensure 'Date' column is datetime
    data_to_process["Date"] = pd.to_datetime(data_to_process["Date"], errors="coerce")
    # Drop rows with missing dates or prices
    clean_data = data_to_process.dropna(subset=["Date", "Close"])

    if clean_data.empty:
        st.error("Dataset has no usable Date/Close columns after cleaning. Please check your data.")
        st.session_state.data_loaded = False # Reset flag as data isn't usable
        st.stop()

    prophet_df_cleaned = (
        clean_data[["Date", "Close"]]
        .rename(columns={"Date": "ds", "Close": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )
    # --- IMPORTANT FIX: Remove timezone from 'ds' column for Prophet ---
    prophet_df_cleaned['ds'] = prophet_df_cleaned['ds'].dt.tz_localize(None)
    # -------------------------------------------------------------------

    st.session_state.prophet_df = prophet_df_cleaned
    # Reset the flag after processing
    st.session_state.data_loaded = False # Reset this flag after processing is done
else:
    prophet_df_cleaned = st.session_state.prophet_df # Use existing processed data

# Now, ensure `clean` is available for pages like `candlestick()`
clean = st.session_state.data.copy() # Use the raw downloaded/uploaded data
clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
clean = clean.dropna(subset=["Date", "Close", "Open", "High", "Low"])


###############################################################################
# Fit Prophet model                                                           #
###############################################################################

# Only re-fit model if data changed or model isn't fitted yet
if st.session_state.model is None or st.session_state.forecast is None:
    with st.spinner("Fitting Prophet model and generating forecast…"):
        try:
            model = Prophet()
            model.fit(prophet_df_cleaned)
            future = model.make_future_dataframe(periods=365) # Forecast 1 year ahead
            forecast = model.predict(future)
            st.session_state.model = model
            st.session_state.forecast = forecast
            st.success("✅ Prophet model fitted and forecast generated!")
        except Exception as e:
            st.error(f"Error fitting Prophet model: {e}")
            st.session_state.model = None
            st.session_state.forecast = None
            st.stop()
else:
    # Use existing model and forecast
    model = st.session_state.model
    forecast = st.session_state.forecast

###############################################################################
# Sidebar navigation                                                          #
###############################################################################

page = st.sidebar.radio(
    "Navigate",
    (
        "Overview",
        "Historical Candlestick",
        "Forecast Line",
        "Actual vs Predicted",
        "Residuals",
        "Components (Year / Week)",
        "Monthly Forecast (12 mo)",
        "Compare Price",
    ),
)

###############################################################################
# Page functions                                                              #
###############################################################################

def overview():
    url = "https://facebook.github.io/prophet/"
    st.markdown(
        f"""
        ### Prophet forecast for `{st.session_state.symbol_title}`  
        **Rows**: {len(clean)}   •   **Date range**: {clean['Date'].min().date()} → {clean['Date'].max().date()}  
        **Model docs**: [Prophet]({url})
        """
    )
    st.dataframe(clean.head())
    st.markdown("""
    **About this page:** This section provides a brief summary of the loaded stock data and the source of the Prophet model.
    It shows the total number of data points and the date range covered, giving you an initial overview of the dataset used for prediction.
    """)


def candlestick():
    st.subheader(f"{st.session_state.symbol_title} – Historical prices")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=clean["Date"],
                open=clean["Open"],
                high=clean["High"],
                low=clean["Low"],
                close=clean["Close"],
                name="OHLC",
            )
        ]
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **About this graph:** A candlestick chart visually represents the Open, High, Low, and Close (OHLC) prices of the stock for each day.
    * **Green/Hollow Candlestick:** Indicates that the closing price was higher than the opening price (a price increase).
    * **Red/Filled Candlestick:** Indicates that the closing price was lower than the opening price (a price decrease).
    * **Wicks (thin lines):** Show the high and low prices reached during the day.
    * **Trend:** By observing patterns in these candlesticks over time, you can identify upward (bullish), downward (bearish), or sideways (ranging) trends. For example, a series of green candlesticks with higher highs and higher lows often indicates an uptrend.
    """)


def forecast_line():
    st.subheader("1-year forecast")
    st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)
    st.markdown("""
    **About this graph:** This plot shows the historical stock prices (`y` values) along with the Prophet model's 1-year forecast (`yhat`).
    * **Black dots:** Represent the actual historical closing prices of the stock.
    * **Blue line:** Represents the Prophet model's predicted future stock prices.
    * **Light blue shaded area:** This is the uncertainty interval (yhat_lower and yhat_upper), showing the range within which the actual future price is expected to fall with a certain probability (default is 80%). A wider interval indicates higher uncertainty in the forecast.
    * **Trend:** You can clearly see the overall trend (upward, downward, or flat) that Prophet has identified in the historical data and extrapolated into the future. This gives a general direction for the stock's expected movement.
    """)


def actual_vs_pred():
    merged = pd.merge(
        prophet_df_cleaned, # Use the cleaned df for actuals
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="inner",
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["ds"], y=merged["y"], name="Actual", mode="lines"))
    fig.add_trace(go.Scatter(x=merged["ds"], y=merged["yhat"], name="Predicted", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=pd.concat([merged["ds"], merged["ds"][::-1]]),
            y=pd.concat([merged["yhat_upper"], merged["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,80,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **About this graph:** This plot directly compares the actual historical stock prices with the Prophet model's predictions for the same historical period.
    * **Blue line:** Represents the **actual** closing prices.
    * **Orange/Red line:** Represents the Prophet model's **predicted** prices (`yhat`).
    * **Shaded area:** The uncertainty interval around the predicted values.
    * **Trend and Accuracy:** Ideally, the "Actual" (blue) line should closely follow the "Predicted" (orange/red) line within the shaded uncertainty band. This helps assess how well the model has learned the historical trend and patterns. Divergence indicates that the model might not be capturing all the nuances of the stock's movement.
    """)

def residuals():
    merged = pd.merge(prophet_df_cleaned, forecast[["ds", "yhat"]], on="ds", how="inner")
    merged["residual"] = merged["y"] - merged["yhat"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Residuals over time")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=merged["ds"], y=merged["residual"], mode="lines", name="Residual"))
        fig1.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("""
        **About 'Residuals over time' graph:** This plot shows the difference between the actual price and the predicted price (the residual) over time.
        * **Points close to zero line:** Indicate good predictions.
        * **Positive residuals:** Model underestimated the actual price.
        * **Negative residuals:** Model overestimated the actual price.
        * **Trend:** Ideally, residuals should be randomly distributed around zero with no clear pattern or trend. Any discernible trend (e.g., consistently positive, consistently negative, increasing/decreasing volatility) suggests that the model might be missing some underlying patterns in the data.
        """)

    with col2:
        st.subheader("Residual distribution")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=merged["residual"], nbinsx=40, name="Histogram"))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
        **About 'Residual distribution' graph:** This histogram visualizes the frequency of different residual values.
        * **Bell-shaped curve centered at zero:** Indicates that the model's errors are normally distributed and unbiased, which is a desirable characteristic.
        * **Skewness or multiple peaks:** Might suggest that the model's assumptions are violated or that there are uncaptured patterns.
        * **Trend:** The shape of this distribution helps in understanding the overall accuracy and bias of the model's predictions. A tight, symmetric distribution around zero means the model is generally accurate.
        """)


def components():
    st.subheader("Prophet components")
    st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)
    st.markdown("""
    **About this graph:** Prophet decomposes the time series into several components, helping you understand the underlying patterns.
    * **Trend:** Shows the overall long-term direction of the stock price, capturing non-periodic changes. A clear upward or downward slope indicates a dominant increasing or decreasing trend.
    * **Yearly Seasonality:** Reveals repetitive patterns that occur over a year. For stocks, this might show typical price movements during certain months or quarters (e.g., year-end rallies, post-earnings dips).
    * **Weekly Seasonality:** Shows repetitive patterns that occur within a week (e.g., how prices typically behave on Mondays vs. Fridays). This is less common for daily stock closing prices but can be present for intraday data.
    * **Trend:** Each component itself represents a trend or pattern at a specific frequency. By analyzing these, you can understand what forces are driving the overall stock price movement. For example, a strong upward trend component indicates fundamental growth, while clear seasonality might point to recurring market behaviors.
    """)


def monthly():
    st.subheader("Monthly forecast (12 months)")
    m = Prophet(changepoint_prior_scale=0.01)
    m.fit(prophet_df_cleaned)
    fut = m.make_future_dataframe(periods=12, freq="MS") # Adjusted frequency to 'MS' for clear monthly starts
    fcst = m.predict(fut)

    fig_monthly = plot_plotly(m, fcst, trend=True, changepoints=True, uncertainty=True,
                              xlabel='Date', ylabel='Predicted Price')

    fig_monthly.add_trace(go.Scatter(x=prophet_df_cleaned['ds'], y=prophet_df_cleaned['y'],
                                     mode='markers', name='Actual Data',
                                     marker=dict(color='blue', size=4)))

    st.plotly_chart(fig_monthly, use_container_width=True)
    st.markdown("""
    **About this graph:** This chart focuses on a 12-month forecast, specifically showing monthly predictions. It uses a slightly different Prophet model configuration (`changepoint_prior_scale=0.01`) which can make the trend more flexible to recent changes.
    * **Blue line:** The monthly predicted stock price.
    * **Shaded area:** The uncertainty interval for the monthly forecast.
    * **Blue markers:** Actual historical data points.
    * **Trend:** This view helps identify shorter-term trends (over the next year) and potential monthly patterns that might not be as apparent in the overall yearly forecast. It can reveal if the stock is expected to have strong or weak months.
    """)


def compare_price():
    st.subheader("Compare Stock Prices")
    st.markdown("""
    **About this page:** This section allows you to compare the historical closing price of the currently loaded stock with another stock.
    This helps in understanding the relative performance and correlation between different assets over the same period.
    """)

    ticker2 = st.text_input("Enter Ticker to Compare (e.g., MSFT)", "MSFT", max_chars=8).upper().strip()
    compare_button = st.button("📈 Compare")

    if compare_button and ticker2:
        if ticker2 == st.session_state.symbol_title:
            st.warning("Please enter a different ticker symbol for comparison.")
            return

        with st.spinner(f"Downloading data for {ticker2}…"):
            try:
                # Use the same date range as the primary stock for fair comparison
                start_compare_date = clean['Date'].min()
                end_compare_date = clean['Date'].max()

                data2 = yf.download(ticker2, start=start_compare_date, end=end_compare_date + pd.Timedelta(days=1))
                data2.reset_index(inplace=True)
                if 'Datetime' in data2.columns: # Handle potential 'Datetime' column from yfinance
                    data2.rename(columns={'Datetime': 'Date'}, inplace=True)
                data2_clean = data2.dropna(subset=["Date", "Close"])

                if data2_clean.empty:
                    st.error(f"No usable data found for {ticker2} in the specified date range.")
                    return

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=clean["Date"], y=clean["Close"], name=st.session_state.symbol_title, mode="lines", line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=data2_clean["Date"], y=data2_clean["Close"], name=ticker2, mode="lines", line=dict(color='red')))

                fig.update_layout(
                    title=f"Price Comparison: {st.session_state.symbol_title} vs {ticker2}",
                    xaxis_title="Date",
                    yaxis_title="Close Price",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"""
                **Analysis of Comparison:** This chart displays the historical closing prices of {st.session_state.symbol_title} (blue line) and {ticker2} (red line).
                * Observe the **relative performance**: Which stock has shown stronger growth or decline over the period?
                * Look for **correlation**: Do the stocks move in the same direction, opposite directions, or are they unrelated? This can indicate if they are influenced by similar or different market forces.
                * Identify **divergence**: Are there periods where one stock significantly outperforms or underperforms the other?
                * **Trend:** You can see if both stocks follow similar general trends (e.g., both in an uptrend) or if their trends diverge. This helps in understanding market sentiment across different sectors or companies.
                """)

            except Exception as e:
                st.error(f"Could not compare {ticker2}: {e}. Check the ticker symbol or internet connection.")
    else:
        st.info("Enter a ticker symbol and click 'Compare' to view the comparison graph.")


###############################################################################
# Page routing                                                                #
###############################################################################

if page == "Overview":
    overview()
elif page == "Historical Candlestick":
    candlestick()
elif page == "Forecast Line":
    forecast_line()
elif page == "Actual vs Predicted":
    actual_vs_pred()
elif page == "Residuals":
    residuals()
elif page == "Components (Year / Week)":
    components()
elif page == "Monthly Forecast (12 mo)":
    monthly()
elif page == "Compare Price":
    compare_price()
