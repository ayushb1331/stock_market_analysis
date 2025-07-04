# 📈 Stock Price Prediction Web App

## 🔍 Overview

A simple web app to predict future stock prices using Python and a tool called Prophet. It’s built with Streamlit, so you can run it in your browser.

## 🧠 Forecasting Model

This app uses the [Facebook Prophet](https://facebook.github.io/prophet/) forecasting model. Prophet is robust, easy to use, and handles:

- Seasonality (weekly/monthly)
- Trend changes
- Missing data
- Outliers

Prophet is well-suited for business forecasting, making it a great educational tool.

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/Uvais5/Stock_Price_Prediction_app.git
cd Stock_Price_Prediction_app

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
