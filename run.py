import yfinance as yf
import talib as ta
import joblib
import pandas as pd
import numpy as np
import index



# ------------------------------
# Helper function to classify candles
# ------------------------------
def candle_type(row):
    """Returns 1 for bullish, -1 for bearish, 0 for doji"""
    if row['Open'] < row['Close']:
        return 1
    elif row['Open'] > row['Close']:
        return -1
    return 0


# ------------------------------
# Download historical data
# ------------------------------
def get_data(tickers, period='210d'):
    """Download historical OHLCV data from Yahoo Finance for the specified tickers"""
    data = {}

    print("Downloading data from Yahoo Finance...")

    for ticker in tickers:
        t = yf.Ticker(ticker)
        try:
            df = t.history(period=period)

            if df.empty:
                print(f"\033[91m[WARNING]\033[0m No data found for {ticker}. Skipping...")
                continue

            df['Type'] = df.apply(candle_type, axis=1)
            data[ticker] = df
            print(f"\033[92m[SUCCESS]\033[0m Data for {ticker} downloaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to download data for {ticker}: {e}")
            continue

    return data


# ------------------------------
# Calculate technical indicators
# ------------------------------
def calculate_indicators(df, params=None):
    """Add multiple technical indicators to a DataFrame"""
    if params is None:
        params = {
            'WilliamsR': 14,
            'ROC': 10,
            'ADX': 14,
            'MACD': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
            'ATR': 14,
            'MFI': 14,
            'STOCH': {'fastk_period': 14, 'slowk_period': 3, 'slowk_matype': 0,
                      'slowd_period': 3, 'slowd_matype': 0}
        }

    # Williams %R
    df['WilliamsR'] = ta.WILLR(df['High'], df['Low'], df['Close'], timeperiod=params['WilliamsR'])
    # ROC
    df['ROC'] = ta.ROC(df['Close'], timeperiod=params['ROC'])
    # ADX
    df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=params['ADX'])
    # MACD
    macd, macdsignal, macdhist = ta.MACD(
        df['Close'],
        fastperiod=params['MACD']['fastperiod'],
        slowperiod=params['MACD']['slowperiod'],
        signalperiod=params['MACD']['signalperiod']
    )
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist
    # MFI
    df['MFI'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=params['MFI'])
    # Stochastic
    slowk, slowd = ta.STOCH(
        df['High'], df['Low'], df['Close'],
        fastk_period=params['STOCH']['fastk_period'],
        slowk_period=params['STOCH']['slowk_period'],
        slowk_matype=params['STOCH']['slowk_matype'],
        slowd_period=params['STOCH']['slowd_period'],
        slowd_matype=params['STOCH']['slowd_matype']
    )
    df['STOCH_slowk'] = slowk
    df['STOCH_slowd'] = slowd
    # ATR
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=params['ATR'])
    # SMA
    df['SMA_20'] = ta.SMA(df['Close'], timeperiod=20)
    df['SMA_100'] = ta.SMA(df['Close'], timeperiod=100)

    return df


# ------------------------------
# Detect bullish engulfing patterns
# ------------------------------
def detect_bullish_engulfing(data):
    """Return only tickers that have a bullish engulfing pattern in the latest candle"""
    print("Detecting bullish engulfing patterns...\n")
    filtered_data = {}

    for ticker, df in data.items():
        engulfing = ta.CDLENGULFING(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
        if engulfing[-1] > 0:
            df = calculate_indicators(df)
            filtered_data[ticker] = df

    return filtered_data


# ------------------------------
# Transform DataFrame for model input
# ------------------------------
def transform_df(df):
    """Prepare a single-row DataFrame with features for the ML model"""
    row = df.iloc[-1]
    prev_row = df.iloc[-2]
    row_sma = df.iloc[-10]

    if prev_row['SMA_20'] < row_sma['SMA_20']:
        new_df = pd.DataFrame([{
            'pOpen': prev_row['Open'],
            'pHigh': prev_row['High'],
            'pClose': prev_row['Close'],
            'pLow': prev_row['Low'],
            'cOpen': row['Open'],
            'cHigh': row['High'],
            'cClose': row['Close'],
            'cLow': row['Low'],
            'williamsR': prev_row['WilliamsR'],
            'roc': prev_row['ROC'],
            'adx': prev_row['ADX'],
            'macdHist': prev_row['MACD_Hist'],
            'mfi': prev_row['MFI'],
            'stoch_slowd': prev_row['STOCH_slowd'],
            'atr': prev_row['ATR'],
            'smaSignal': prev_row['SMA_100'] / row['Close'],
        }])
        return new_df
    else:
        # Return empty DataFrame if condition not met
        return pd.DataFrame()


# ------------------------------
# Main model execution
# ------------------------------
def run_model(threshold=0.52):
    print("Running Bullish Engulfing Model... 🚀\n")
    
    # 1. Get historical data
    data = get_data(index.TICKERS_NASDAQ100)

    # 2. Filter tickers with bullish engulfing
    data = detect_bullish_engulfing(data)

    data_model = {}

    # 3. Transform each DataFrame to model input
    for ticker, tick_df in data.items():
        tick_df = calculate_indicators(tick_df).dropna()
        dm = transform_df(tick_df)
        if not dm.empty:
            data_model[ticker] = dm

    # 4. Load scaler, PCA, and ML model
    scaler = joblib.load('bullScaler.pkl')
    pca = joblib.load('bullPCA.pkl')
    model = joblib.load('modelBullEngulf_03.09.2025.joblib')

    # 5. Prepare features for model
    features_original = ['pOpen','pHigh','pClose','pLow','cOpen','cHigh','cClose','cLow']
    features_model = ['williamsR','adx','macdHist','roc','atr','mfi','stoch_slowd','smaSignal',
                      'PCA1','PCA2','PCA3','PCA4']

    # 6. Apply scaling, PCA, and make predictions
    for ticker, tick_df in data_model.items():
        # Scale original columns
        X_scaled = scaler.transform(tick_df[features_original].values)
        # PCA transformation
        components = pca.transform(X_scaled)
        tick_df['PCA1'] = components[:, 0]
        tick_df['PCA2'] = components[:, 1]
        tick_df['PCA3'] = components[:, 2]
        tick_df['PCA4'] = components[:, 3]

        # Model input (only features used in training)
        X_model = tick_df[features_model]
        probs = model.predict_proba(X_model)[:, 1]  # probability of True
        predictions = probs >= threshold  # apply custom threshold

        # Print result
        print(f"Bullish Engulfing in {ticker}:")
        if predictions[0]== True:
            txt = f"Predict: \033[92m{predictions[0]}\033[0m \tProb: {probs[0]:.3f}\n"
        else:
            txt = f"\033[91mPredict: \033[91m{predictions[0]}\n\033[0m \tProb: {probs[0]:.3f}"

        print(txt)


# ------------------------------
# Execute main
# ------------------------------
if __name__ == "__main__":
    run_model()