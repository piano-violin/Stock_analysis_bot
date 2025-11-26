# Stock_analysis_bot
# Stock Analysis Telegram Bot with ML forecasting

Проект направлен на разработку телеграм-бота, который позволяет пользователюполучать прогноз цен акций и рекомендации по торговым стратегиям. Пользователь вводит название компании и сумму для условной инвестиции, ботавтоматически загружает исторические данные о стоимости акций, обучаетнесколько моделей временных рядов, выбирает наилучшую по метрикам качестваи строит прогноз на ближайшие 30 дней.
В финале пользователь получает: 
- график прогноза;
- оценку изменения цены акций относительно текущего дня;
- сводку с рекомендациями по дням покупки и продажи;
- расчет потенциальной прибыли.
Дополнительно бот ведет журнал логов, где фиксируются все ключевые параметры каждого запроса.

# 🤖 Stock Analysis Telegram Bot

Telegram bot for stock price forecasting and trading recommendations using machine learning and time series analysis.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Telegram](https://img.shields.io/badge/Telegram-Bot-blue)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest%2C%20LSTM%2C%20ARIMA-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 📊 Overview

This Telegram bot analyzes stock prices using three different machine learning models, selects the best performing one, and provides trading recommendations with profit calculations. The bot automatically downloads historical data, trains multiple models, and generates 30-day forecasts.

## ✨ Features

- **📈 Multi-model Forecasting**: Implements Random Forest, ARIMA, and LSTM models
- **🤖 Automatic Model Selection**: Chooses best model based on MAPE metric
- **📅 30-Day Business Forecast**: Predicts stock prices for next 30 working days
- **💼 Trading Recommendations**: Identifies optimal buy/sell days with profit calculation
- **📊 Interactive Visualization**: Generates historical + forecast plots
- **📝 Comprehensive Logging**: Tracks all user sessions and model performance
- **🔧 Error Handling**: Robust error handling for invalid inputs

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Telegram Bot Token from [@BotFather](https://t.me/BotFather)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/stock-analysis-bot.git
cd stock-analysis-bot
