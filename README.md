# 📈 Bullish Engulfing Pattern Prediction

This project aims to evaluate the effectiveness of the **bullish engulfing candlestick pattern** using machine learning models. The goal is to predict whether the pattern will lead to a positive price movement within 15 days.

## Features

- Historical stock data collection from Yahoo Finance
- Technical indicators: RSI, Williams %R, ADX, MACD, Bollinger Bands, ATR
- Pattern detection: Bullish Engulfing
- Feature engineering including PCA for correlated price variables
- Models implemented:
  - Random Forest
  - XGBoost
  - Neural Networks
- Threshold analysis to optimize model performance
- Model evaluation with confusion matrices, ROC curves, and key metrics (accuracy, precision, recall, F1-score)

## Notes

- The predictive models are tools for analysis, not trading recommendations.
- Results should be combined with fundamental analysis and expert judgment when making investment decisions.

## License

This project is licensed under the MIT License.


---

## ⚡ Features

- Downloads stock data from **Yahoo Finance**  
- Detects **Bullish Engulfing** patterns using **TA-Lib**  
- Calculates multiple technical indicators (ADX, ROC, ATR, MFI, Stochastic, etc.)  
- Applies **PCA** for dimensionality reduction  
- Uses a trained **Random Forest model** to predict signals  

---

## 📌 Usage

This project provides an executable script `run.py` that downloads market data, detects **Bullish Engulfing** patterns, applies technical indicators, and runs a trained ML model to predict the probability of confirmation.

### ▶️ Running the script

From the project root, simply run:

```bash
python run.py
```

If conditions are met, the script will print predictions like:

```
Ejecutando modelos... 🚀
Bullish Engulfing in WBD:
Predict: True    Prob: 0.5467
```

- **Predict** → Model’s decision (True/False)  
- **Prob** → Probability of the bullish engulfing confirmation  

---

## 📦 Requirements

Install the required libraries before running the script:

```bash
pip install yfinance ta-lib scikit-learn pandas numpy joblib
```
---

## ⚙️ Dependencies

- **[yfinance](https://pypi.org/project/yfinance/)** → stock market data  
- **[TA-Lib](https://mrjbq7.github.io/ta-lib/)** → technical indicators & candlestick patterns  
- **pandas** → data manipulation  
- **numpy** → numerical operations  
- **scikit-learn** → machine learning (RandomForest, PCA, Scaler, etc.)  
- **joblib** → load scaler, PCA, and trained model files  

---

## 📂 Required Files

Make sure the following files are present in your project directory:

- `bullScaler.pkl` → trained scaler  
- `bullPCA.pkl` → trained PCA  
- `modelBullEngulf_xxx.joblib` → trained ML model  

---

## 🚀 Roadmap

- [ ] Add support for more candlestick patterns  
- [ ] Implement backtesting module  
- [ ] Deploy as a REST API  

---

## 📝 License

This project is licensed under the MIT License.
