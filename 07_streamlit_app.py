import streamlit as st
import pandas as pd

# Dashboard Title
st.title("📈 Stock Prediction & Risk Dashboard")
st.markdown("""
### 📌 Project Overview

This dashboard combines:

- Stock Price Prediction using Machine Learning
- Risk Analysis using Volatility, Sharpe Ratio & Drawdown
- Stock Risk Index (SRI)
- Buy/Hold/Sell Recommendation Engine
""")

# Load Data
master_df = pd.read_csv(
    "outputs/master_df.csv"
)

master_df['Date'] = pd.to_datetime(
    master_df['Date']
)
risk_df = pd.read_csv("./outputs/risk_df.csv")

# Show Data
st.subheader("Master Data")
st.dataframe(master_df.head())

st.subheader("Risk Summary")
st.dataframe(risk_df)
# KPI Section
st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

avg_return = risk_df['Annual_Return'].mean()

avg_risk = risk_df['Annualized_Volatility'].mean()

best_stock = risk_df.loc[
    risk_df['Annual_Return'].idxmax(),
    'Company'
]

lowest_risk_stock = risk_df.loc[
    risk_df['Annualized_Volatility'].idxmin(),
    'Company'
]

col1.metric(
    "Avg Return",
    f"{avg_return:.2%}"
)

col2.metric(
    "Avg Risk",
    f"{avg_risk:.2%}"
)

col3.metric(
    "Best Stock",
    best_stock
)

col4.metric(
    "Lowest Risk",
    lowest_risk_stock
)
# Stock Selector
st.subheader("📌 Select Stock")

selected_stock = st.selectbox(
    "Choose a Stock",
    master_df['Company'].unique()
)

st.write("Selected Stock:", selected_stock)


import plotly.express as px

st.subheader("📈 Stock Price Trend")

stock_data = master_df[
    master_df['Company'] == selected_stock
]

fig = px.line(
    stock_data,
    x='Date',
    y='Close',
    title=f'{selected_stock} Closing Price'
)

st.plotly_chart(
    fig,
    use_container_width=True
)

#Return Distribution
st.subheader("📊 Return Distribution")

# Metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Avg Daily Return",
    f"{stock_data['Daily_Return'].mean()*100:.2f}%"
)

col2.metric(
    "Daily Volatility",
    f"{stock_data['Daily_Return'].std()*100:.2f}%"
)

col3.metric(
    "Positive Days",
    int((stock_data['Daily_Return'] > 0).sum())
)

col4.metric(
    "Negative Days",
    int((stock_data['Daily_Return'] < 0).sum())
)

# Histogram
avg_return = stock_data['Daily_Return'].mean()

fig = px.histogram(
    stock_data,
    x="Daily_Return",
    nbins=30,
    title=f"{selected_stock} Daily Return Distribution"
)

# Average return line
fig.add_vline(
    x=avg_return,
    line_dash="dash",
    annotation_text="Average Return"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Simple explanation
st.info(
    f"""
    {selected_stock} generates an average daily return of
    {avg_return*100:.2f}% with a daily volatility of
    {stock_data['Daily_Return'].std()*100:.2f}%.
    
    Positive Days: {(stock_data['Daily_Return'] > 0).sum()}
    
    Negative Days: {(stock_data['Daily_Return'] < 0).sum()}
    """
)

#Risk vs Return Analysis
st.header("📈 Risk vs Return Analysis")

fig = px.scatter(
    risk_df,
    x="Annualized_Volatility",
    y="Annual_Return",
    text="Company",
    size="SRI",
    hover_name="Company",
    color="Risk_Level"
)

fig.update_traces(
    textposition="top center"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


#Step 9: Load Your Trained Model
import joblib
model = joblib.load("outputs/models/best_stock_prediction_model.pkl")

st.header("🔮 Tomorrow Price Prediction")
latest_data = (
    master_df[master_df["Company"] == selected_stock]
    .sort_values("Date")
    .tail(1)
)
features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA_7",
    "MA_21",
    "MA_50",
    "Lag_1",
    "Lag_5",
    "Lag_10",
    "Rolling_Mean_7",
    "Rolling_Mean_21",
    "Rolling_Std_7",
    "Rolling_Std_21"
]
prediction = model.predict(
    latest_data[features]
)[0]
current_price = latest_data["Close"].values[0]

change_pct = (
    (prediction - current_price)
    / current_price
) * 100
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Current Price",
        f"₹{current_price:,.2f}"
    )

with col2:
    st.metric(
        "Predicted Tomorrow",
        f"₹{prediction:,.2f}"
    )

with col3:
    st.metric(
        "Expected Change",
        f"{change_pct:.2f}%"
    )
    
# =====================================
# Additional Prediction Insights
# =====================================

price_diff = prediction - current_price

st.subheader("📊 Prediction Insights")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Expected Move",
        f"₹{price_diff:.2f}"
    )

with col2:
    st.metric(
        "Expected Return",
        f"{change_pct:.2f}%"
    )

# Buy / Hold / Sell Recommendation

if change_pct > 1:
    recommendation = "🟢 Strong Buy"
elif change_pct > 0:
    recommendation = "🔵 Buy"
elif change_pct > -1:
    recommendation = "🟡 Hold"
else:
    recommendation = "🔴 Sell"

st.info(f"📌 Recommendation: {recommendation}")

# Bullish / Bearish Signal

if prediction > current_price:
    st.success(
        "📈 Bullish Signal: Model expects price to increase tomorrow."
    )
else:
    st.error(
        "📉 Bearish Signal: Model expects price to decrease tomorrow."
    )
    
#SRI Comparison Chart
    # =====================================
# Stock Risk Index Comparison
# =====================================

st.header("⚠️ Stock Risk Index (SRI) Comparison")

import plotly.express as px

fig = px.bar(
    risk_df,
    x="Company",
    y="SRI",
    color="Risk_Level",
    text="SRI",
    title="Stock Risk Index of Companies"
)

fig.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Company",
    yaxis_title="Stock Risk Index (SRI)",
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.info("""
📌 Higher SRI = Higher Risk

• Low SRI → Safer Stock
• Medium SRI → Moderate Risk
• High SRI → Aggressive Risk
""")
st.markdown("---")
st.caption(
    "Developed by Deepak Das | Stock Price Prediction & Risk Dashboard"
)