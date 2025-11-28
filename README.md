# 🤖 Stock Analysis Telegram Bot with ML forecasting / Телеграм-бот для анализа акций 


Образовательный проект направлен на разработку телеграм-бота, который позволяет пользователюполучать прогноз цен акций и рекомендации по торговым стратегиям. Пользователь вводит название компании и сумму для условной инвестиции, ботавтоматически загружает исторические данные о стоимости акций, обучаетнесколько моделей временных рядов, выбирает наилучшую по метрикам качестваи строит прогноз на ближайшие 30 дней.
В финале пользователь получает: 
- график прогноза;
- оценку изменения цены акций относительно текущего дня;
- сводку с рекомендациями по дням покупки и продажи;
- расчет потенциальной прибыли.
Дополнительно бот ведет журнал логов, где фиксируются все ключевые параметры каждого запроса.


![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![Telegram](https://img.shields.io/badge/Telegram-Bot-blue)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest%2C%20LSTM%2C%20ARIMA-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 📊 Overview / Обзор

This Telegram bot analyzes stock prices using three different machine learning models, selects the best performing one, and provides trading recommendations with profit calculations. The bot automatically downloads historical data, trains multiple models, and generates 30-day forecasts.

Этот Telegram бот анализирует цены акций с использованием трех различных моделей машинного обучения, выбирает лучшую из них и предоставляет торговые рекомендации с расчетом прибыли. Бот автоматически загружает исторические данные, обучает несколько моделей и генерирует 30-дневные прогнозы.

## ✨ Features / Возможности

- **📈 Multi-model Forecasting**: Implements Random Forest, ARIMA, and LSTM models
- **🤖 Automatic Model Selection**: Chooses best model based on MAPE metric
- **📅 30-Day Business Forecast**: Predicts stock prices for next 30 working days
- **💼 Trading Recommendations**: Identifies optimal buy/sell days with profit calculation
- **📊 Interactive Visualization**: Generates historical + forecast plots
- **📝 Comprehensive Logging**: Tracks all user sessions and model performance
- **🔧 Error Handling**: Robust error handling for invalid inputs

`

- **📈 Многомодельное прогнозирование**: Реализованы модели Random Forest, ARIMA и LSTM
- **🤖 Автоматический выбор модели**: Выбирает лучшую модель на основе метрики MAPE
- **📅 30-дневный бизнес-прогноз**: Предсказывает цены акций на следующие 30 рабочих дней
- **💼 Торговые рекомендации**: Определяет оптимальные дни для покупки/продажи с расчетом прибыли
- **📊 Интерактивная визуализация**: Создает графики исторических данных + прогнозов
- **📝 Комплексное логирование**: Отслеживает все пользовательские сессии и производительность моделей
- **🔧 Обработка ошибок**: Надежная обработка неверных входных данных

## 🚀 Quick Start / Быстрый старт

### Prerequisites / Предварительные требования

- Python 3.13 or higher
- Telegram Bot Token from [@BotFather](https://t.me/BotFather)

## 💡 How to Use / Как использовать
Start your bot in Telegram and use these commands / Запустите вашего бота в Telegram и используйте следующие команды:

Basic Commands / Основные команды:
- `/start` or `/help` - Show help message and instructions / Показать справочное сообщение и инструкции
- `/test TICKER` - Test data loading for a specific ticker / Протестировать загрузку данных для конкретного тикера
- `/exit` - Exit the bot / Выйти из бота

## 🛠️ Technical Implementation / Техническая реализация

### Machine Learning Models / Модели машинного обучения

| Model | Type | Configuration | Features |
|-------|------|---------------|----------|
| **Random Forest** | Ensemble | 50 estimators, 15-day lags | Lagged price features |
| **ARIMA** | Statistical | (2,1,2) order | Auto-regressive |
| **LSTM** | Neural Network | 50 units, 30-day lookback | Sequence learning |

### Model Selection Criteria
- **Primary Metric / Основная метрика**: MAPE (Mean Absolute Percentage Error)
- **Secondary Metric / Вторичная метрика**: RMSE (Root Mean Square Error)
- **Validation / Валидация**: 80/20 train-test split on temporal data

## Data Pipeline / Пайплайн данных
- **Data Source / Источник данных**: Yahoo Finance via `yfinance` API
- **Time Period / Временной период**: 2 years of historical data / 2 года исторических данных
- **Frequency / Частота**: Business days only (Monday-Friday) / Только рабочие дни
- **Preprocessing / Предобработка**: Automatic missing value handling / Автоматическая обработка пропущенных значений

## **📊 Supported Tickers / Поддерживаемые тикеры**

The bot works with any valid Yahoo Finance ticker / Бот работает с любыми тикерами Yahoo Finance:
- US Stocks: AAPL (Apple), TSLA (Tesla), GOOGL (Google), MSFT (Microsoft)
- ETFs: SPY (S&P 500), QQQ (Nasdaq 100)

Find more tickers at / Полный список тикеров: [Yahoo Finance](https://finance.yahoo.com/)

## **⚠️ Disclaimer / Отказ от ответственности**
<div align="center">

### **IMPORTANT: This is an EDUCATIONAL PROJECT only. Not a real trading tool.**
  
### **ВАЖНО: Это исключительно ОБРАЗОВАТЕЛЬНЫЙ ПРОЕКТ. Не является реальным торговым инструментом.**

</div>
