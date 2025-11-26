import pandas as pd
import telebot
import yfinance as yf
import datetime
import numpy as np
# Важно! Импортируем matplotlib до pyplot, иначе будут возникать проблемы при отрисовке графиков
import matplotlib
matplotlib.use('Agg')  # Неинтерактивный бэкенд

import matplotlib.pyplot as plt
import io
import logging
import traceback
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    filename='logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

API_TOKEN = 'add your token here'
bot = telebot.TeleBot(API_TOKEN)

def get_business_days_count(start_date, end_date):
    """Рассчитывает количество рабочих дней между двумя датами."""
    return len(pd.bdate_range(start=start_date, end=end_date))

def generate_future_business_days(last_date, days=30):
    """Генерирует даты следующих 30 рабочих дней."""
    future_dates = []
    current_date = last_date + datetime.timedelta(days=1)
    
    while len(future_dates) < days:
        if current_date.weekday() < 5:  # Понедельник-Пятница
            future_dates.append(current_date)
        current_date += datetime.timedelta(days=1)
    
    return future_dates[:days]

def load_data(company):
    """Загружает исторические данные за последние 2 года с учетом рабочих дней."""
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=730)  # Примерно 2 года
        
        print(f"Загрузка данных для {company} с {start.date()} по {end.date()}")
        
        data = yf.download(company, start=start, end=end, auto_adjust=True)
        
        if data.empty:
            raise ValueError(f"Данные для тикера {company} не найдены.")
        
        # Проверяем и нормализуем название столбца с ценами закрытия
        if 'Close' in data.columns:
            close_data = data['Close']
        elif 'Adj Close' in data.columns:
            close_data = data['Adj Close']
            print("Используется столбец 'Adj Close'")
        else:
            # Ищем любой столбец с ценами
            price_columns = [col for col in data.columns if 'close' in col.lower() or 'price' in col.lower()]
            if price_columns:
                close_data = data[price_columns[0]]
                print(f"Используется столбец '{price_columns[0]}'")
            else:
                raise ValueError("Не найден столбец с ценами закрытия")
        
        # Убедимся, что у нас только рабочие дни
        close_data = close_data.asfreq('B')  # Business days
        close_data = close_data.ffill()  # Заполняем пропущенные дни (если есть)
        
        print(f"Загружено {len(close_data)} рабочих дней данных")
        print(f"Диапазон дат: {close_data.index[0].date()} - {close_data.index[-1].date()}")
        
        return close_data
    
    except Exception as e:
        print(f"Ошибка при загрузке данных: {str(e)}")
        raise

def prepare_train_test_data(data, test_size=0.2):
    """Подготавливает обучающую и тестовую выборки с учетом временных рядов."""
    split_idx = int(len(data) * (1 - test_size))
    train = data[:split_idx]
    test = data[split_idx:]
    
    # Убедимся, что между train и test нет разрыва в датах
    expected_next_date = train.index[-1] + pd.offsets.BDay(1)
    if test.index[0] != expected_next_date:
        print(f"Предупреждение: разрыв в датах между train и test")
    
    return train, test

# 1. Функции для RandomForest
def create_lagged_features(data, n_lags=15):
    """Создает лаговые признаки"""
    try:
        # Преобразуем в 1D массив если нужно
        if hasattr(data, 'values'):
            values = data.values.flatten()
        else:
            values = np.array(data).flatten()
        
        # Создаем простой DataFrame
        df = pd.DataFrame({'Close': values})
        
        # Только основные лаги
        for i in range(1, n_lags + 1):
            df[f'lag_{i}'] = df['Close'].shift(i)
        
        return df.dropna()
    except Exception as e:
        print(f"Ошибка создания признаков: {e}")
        raise

def train_random_forest(train_data, test_data):
    """Обучает модель Random Forest."""
    try:
        n_lags = 15
        
        # Подготавливаем данные для обучения
        df_train = create_lagged_features(train_data, n_lags)
        df_test = create_lagged_features(test_data, n_lags)
        
        if len(df_train) < 5 or len(df_test) < 2:
            raise ValueError("Недостаточно данных после создания признаков")
        
        X_train = df_train.drop('Close', axis=1)
        y_train = df_train['Close']
        X_test = df_test.drop('Close', axis=1)
        y_test = df_test['Close']
        
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return model, rmse, mape, y_pred, n_lags
        
    except Exception as e:
        print(f"Ошибка в Random Forest: {str(e)}")
        raise

# 2. Функция для ARIMA
def train_arima(train_data, test_data):
    """Обучает модель ARIMA."""
    try:
        # Используем автоматический подбор параметров
        model = ARIMA(train_data, order=(2, 1, 2))
        fitted_model = model.fit()
        
        # Прогноз на длину тестовой выборки
        forecast = fitted_model.forecast(steps=len(test_data))
        y_pred = forecast.values
        
        rmse = np.sqrt(mean_squared_error(test_data, y_pred))
        mape = mean_absolute_percentage_error(test_data, y_pred)
        
        return fitted_model, rmse, mape, y_pred
        
    except Exception as e:
        print(f"Ошибка в ARIMA: {str(e)}")
        # Возвращаем простую модель в случае ошибки
        try:
            model = ARIMA(train_data, order=(1, 1, 1))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=len(test_data))
            y_pred = forecast.values
            
            rmse = np.sqrt(mean_squared_error(test_data, y_pred))
            mape = mean_absolute_percentage_error(test_data, y_pred)
            
            return fitted_model, rmse, mape, y_pred
        except:
            # Если ARIMA полностью не работает:
            y_pred = np.full(len(test_data), test_data.mean())
            rmse = np.sqrt(mean_squared_error(test_data, y_pred))
            mape = mean_absolute_percentage_error(test_data, y_pred)
            return None, rmse, mape, y_pred

# 3. Функция для LSTM 
def create_lstm_dataset(data, lookback=30):
    """Создает датасет для LSTM."""
    X, y = [], []
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i - lookback:i])
        y.append(data_scaled[i])
    
    return np.array(X), np.array(y), scaler

def train_lstm(train_data, test_data, lookback=30):
    """Обучает модель LSTM."""
    try:
        # Подготовка обучающих данных
        X_train, y_train, scaler = create_lstm_dataset(train_data, lookback)
        
        # Подготовка тестовых данных
        full_series = pd.concat([train_data, test_data])
        X_test_full, y_test_full, _ = create_lstm_dataset(full_series, lookback)
        
        # Берем только часть, соответствующую тестовому периоду
        test_start_idx = len(train_data) - lookback
        X_test = X_test_full[test_start_idx:test_start_idx + len(test_data)]
        y_test = y_test_full[test_start_idx:test_start_idx + len(test_data)]
        
        # Преобразование для LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Создаем упрощенную модель
        model = Sequential([
            LSTM(50, return_sequences=False, input_shape=(lookback, 1)),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=0)
        
        # Прогноз
        test_predict = model.predict(X_test, verbose=0)
        test_predict_inv = scaler.inverse_transform(test_predict).flatten()
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
        mape = mean_absolute_percentage_error(y_test_inv, test_predict_inv)
        
        return model, rmse, mape, test_predict_inv, scaler, lookback
        
    except Exception as e:
        print(f"Ошибка в LSTM: {str(e)}")
        raise

# 4. Функция для выбора лучшей модели
def select_best_model(company):
    """Выбирает лучшую модель для прогнозирования."""
    try:
        # Загрузка данных
        data = load_data(company)
        
        if len(data) < 100:
            raise ValueError(f"Недостаточно данных. Требуется минимум 100 рабочих дней, получено {len(data)}")
        
        # Подготовка данных
        train_data, test_data = prepare_train_test_data(data)
        
        print(f"Обучающая выборка: {len(train_data)} дней")
        print(f"Тестовая выборка: {len(test_data)} дней")
        
        models_results = {}
        
        # Обучение Random Forest
        try:
            rf_model, rf_rmse, rf_mape, rf_pred, n_lags = train_random_forest(train_data, test_data)
            models_results['Random Forest'] = {
                'model': rf_model, 
                'rmse': rf_rmse, 
                'mape': rf_mape,
                'n_lags': n_lags
            }
            print(f"✅ Random Forest: MAPE = {rf_mape:.4f}")
        except Exception as e:
            print(f"❌ Random Forest: {e}")
            models_results['Random Forest'] = {'model': None, 'mape': float('inf')}
        
        # Обучение ARIMA
        try:
            arima_model, arima_rmse, arima_mape, arima_pred = train_arima(train_data, test_data)
            models_results['ARIMA'] = {
                'model': arima_model, 
                'rmse': arima_rmse, 
                'mape': arima_mape
            }
            print(f"✅ ARIMA: MAPE = {arima_mape:.4f}")
        except Exception as e:
            print(f"❌ ARIMA: {e}")
            models_results['ARIMA'] = {'model': None, 'mape': float('inf')}
        
        # Обучение LSTM
        try:
            lstm_model, lstm_rmse, lstm_mape, lstm_pred, scaler, lookback = train_lstm(train_data, test_data)
            models_results['LSTM'] = {
                'model': lstm_model, 
                'rmse': lstm_rmse, 
                'mape': lstm_mape,
                'scaler': scaler,
                'lookback': lookback
            }
            print(f"✅ LSTM: MAPE = {lstm_mape:.4f}")
        except Exception as e:
            print(f"❌ LSTM: {e}")
            models_results['LSTM'] = {'model': None, 'mape': float('inf')}
        
        # Выбор лучшей модели
        valid_models = {name: metrics for name, metrics in models_results.items() 
                       if metrics['model'] is not None and metrics['mape'] < float('inf')}
        
        if not valid_models:
            raise ValueError("Ни одна модель не была успешно обучена")
        
        best_model_name = min(valid_models, key=lambda x: valid_models[x]['mape'])
        best_model_info = valid_models[best_model_name]
        
        print(f"🏆 Лучшая модель: {best_model_name} (MAPE: {best_model_info['mape']:.4f})")
        
        return best_model_name, best_model_info, data
        
    except Exception as e:
        print(f"Ошибка при выборе модели: {e}")
        raise

# 5. Функция для построения прогноза
def make_forecast(best_model_name, best_model_info, historical_data, days=30):
    """Строит прогноз на указанное количество рабочих дней."""
    try:
        future_predictions = []
        last_date = historical_data.index[-1]
        future_dates = generate_future_business_days(last_date, days)
        
        data = historical_data.copy()
        
        if best_model_name == 'Random Forest':
            n_lags = best_model_info['n_lags']
            model = best_model_info['model']
            
            # предсказываем все сразу на основе последних доступных данных
            last_features = create_lagged_features(data, n_lags).iloc[-1:].drop('Close', axis=1)
            
            # Предсказываем все 30 дней сразу
            for i in range(days):
                next_pred = model.predict(last_features)[0]
                future_predictions.append(next_pred)
                
                # Обновляем признаки для следующего шага (сдвигаем окно)
                last_features_values = last_features.values[0]
                new_features = np.roll(last_features_values, 1)
                new_features[0] = next_pred
                last_features = pd.DataFrame([new_features], columns=last_features.columns)

        
        elif best_model_name == 'ARIMA':
            model = best_model_info['model']
            forecast = model.forecast(steps=days)
            future_predictions = forecast.values.tolist()
        
        elif best_model_name == 'LSTM':
            model = best_model_info['model']
            scaler = best_model_info['scaler']
            lookback = best_model_info['lookback']
            
            # Берем последние lookback значений
            last_sequence = scaler.transform(data.values[-lookback:].reshape(-1, 1)).flatten()
            
            for _ in range(days):
                X_input = last_sequence[-lookback:].reshape(1, lookback, 1)
                next_pred_scaled = model.predict(X_input, verbose=0)[0, 0]
                next_pred = scaler.inverse_transform([[next_pred_scaled]])[0, 0]
                future_predictions.append(next_pred)
                last_sequence = np.append(last_sequence, next_pred_scaled)
        
        # Создаем DataFrame с прогнозом
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_predictions
        })
        forecast_df.set_index('Date', inplace=True)
        
        return forecast_df
        
    except Exception as e:
        print(f"Ошибка при построении прогноза: {e}")
        raise

# 6. Функция для визуализации
def create_plot(historical_data, forecast_df, company):
    """Создает график исторических данных и прогноза."""
    plt.figure(figsize=(12, 6))
    
    # Исторические данные
    plt.plot(historical_data.index, historical_data.values, 
             label='Исторические данные', color='blue', linewidth=2)
    
    # Прогноз
    plt.plot(forecast_df.index, forecast_df['Predicted_Close'], 
             label='Прогноз на 30 рабочих дней', color='red', linestyle='--', linewidth=2)
    
    # Вертикальная линия разделения
    last_historical_date = historical_data.index[-1]
    plt.axvline(x=last_historical_date, color='gray', linestyle=':', alpha=0.7)
    
    plt.title(f'Прогноз цены акций {company}\n(рабочие дни)', fontsize=14, fontweight='bold')
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Цена закрытия ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Сохраняем в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

# 7. Генерация торговых рекомендаций и расчета прибыли
def generate_trading_recommendations(forecast_df, investment_amount):
    """Генерирует торговые рекомендации на основе прогноза."""
    prices = forecast_df['Predicted_Close'].values
    
    # Находим локальные минимумы и максимумы
    buy_days = []
    sell_days = []
    
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            buy_days.append(i)
        elif prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            sell_days.append(i)
    
    # Моделируем торговую стратегию
    cash = investment_amount
    shares = 0
    trades = []
    
    # Простая стратегия: покупаем на минимумах, продаем на максимумах
    all_days = list(range(len(prices)))
    actions = ['hold'] * len(prices)
    
    for day in buy_days:
        actions[day] = 'buy'
    for day in sell_days:
        actions[day] = 'sell'
    
    # Симулируем торговлю
    for day, action in enumerate(actions):
        price = prices[day]
        date = forecast_df.index[day]
        
        if action == 'buy' and cash >= price:
            # Покупаем 1 акцию
            shares_to_buy = cash // price
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                shares += shares_to_buy
                cash -= cost
                trades.append(f"{date.strftime('%Y-%m-%d')}: КУПИТЬ {shares_to_buy} акций по ${price:.2f}")
        
        elif action == 'sell' and shares > 0:
            # Продаем все акции
            revenue = shares * price
            cash += revenue
            trades.append(f"{date.strftime('%Y-%m-%d')}: ПРОДАТЬ {shares} акций по ${price:.2f}")
            shares = 0
    
    # Финализируем позицию
    final_value = cash + (shares * prices[-1])
    total_profit = final_value - investment_amount
    profit_percentage = (total_profit / investment_amount) * 100
    
    # Формируем сводку
    summary = f"""
📊 **ИНВЕСТИЦИОННАЯ СВОДКА**

💰 **Начальные инвестиции:** ${investment_amount:,.2f}
🏦 **Финальный капитал:** ${final_value:,.2f}
📈 **Прибыль:** ${total_profit:,.2f} ({profit_percentage:+.2f}%)

🎯 **Рекомендации:**
• Дни для покупки: {[f'День {d+1}' for d in buy_days]}
• Дни для продажи: {[f'День {d+1}' for d in sell_days]}

📅 **Период прогноза:** 30 рабочих дней
"""
    
    return summary, trades, total_profit

@bot.message_handler(commands=['choose'])
def handle_choose(message):
    """Обработчик команды /choose"""
    try:
        parts = message.text.split()
        if len(parts) != 3:
            raise ValueError("Используйте формат: /choose TICKER INVESTMENT")
        
        company = parts[1].upper()
        investment = float(parts[2])
        
        if investment <= 0:
            raise ValueError("Сумма инвестиции должна быть положительной")
        
        # Отправляем сообщение о начале обработки
        processing_msg = bot.send_message(
            message.chat.id, 
            f"🔄 Анализирую {company} с инвестицией ${investment:,.2f}...\nЭто займет 1-2 минуты."
        )
        
        user_id = message.from_user.id
        logging.info(f"UserID: {user_id}, Ticker: {company}, Investment: {investment}")
        
        try:
            # Выбор лучшей модели
            best_model_name, best_model_info, historical_data = select_best_model(company)
            
            # Построение прогноза
            forecast_df = make_forecast(best_model_name, best_model_info, historical_data)
            
            # Создание графика
            plot_buf = create_plot(historical_data, forecast_df, company)
            
            # Генерация рекомендаций
            summary, trades, profit = generate_trading_recommendations(forecast_df, investment)
            
            # Отправка результатов
            bot.send_photo(message.chat.id, plot_buf, 
                         caption=f"📈 Прогноз для {company} на 30 рабочих дней")
            
            bot.send_message(message.chat.id, summary, parse_mode='Markdown')
            
            # Отправка деталей сделок (если есть)
            if trades:
                trades_text = "💼 **Детали сделок:**\n" + "\n".join(trades[:10])  # Ограничиваем вывод
                if len(trades) > 10:
                    trades_text += f"\n... и еще {len(trades) - 10} сделок"
                bot.send_message(message.chat.id, trades_text)
            
            # Логирование успешного завершения
            best_mape = best_model_info['mape']
            logging.info(f"UserID: {user_id}, BestModel: {best_model_name}, "
                        f"MAPE: {best_mape:.4f}, Profit: ${profit:.2f}")
                        
        except Exception as e:
            error_msg = f"❌ Ошибка анализа: {str(e)}"
            bot.send_message(message.chat.id, error_msg)
            logging.error(f"UserID: {user_id}, Error: {str(e)}")
            
        finally:
            # Удаляем сообщение об обработке
            try:
                bot.delete_message(message.chat.id, processing_msg.message_id)
            except:
                pass
                
    except Exception as e:
        bot.send_message(message.chat.id, 
                        f"❌ Ошибка в формате команды: {str(e)}\n"
                        "Пример: `/choose AAPL 10000`", 
                        parse_mode='Markdown')

@bot.message_handler(commands=['start', 'help'])
def handle_start(message):
    """Обработчик команды start"""
    welcome_text = """
🤖 **Бот для анализа акций**

Привет! Я робот на основе нейронной сети и могу помочь управлять портфелем твоих акций (получить прогноз цен акций и рекомендации по торговым стратегиям). 
Для начала нам надо выбрать тикер компании, которая нас интересует, например, для компании APPLE тикером будет AAPL, для Google - GOOGL.
Полный список тикеров можно найти на сайте https://finance.yahoo.com/.
Затем нужно ввести команду /choose и через пробел ввести тикер компании (TICKER). 
Далее еще через пробел ввести сумму для условной инвестиции в виде целого числа (INVESTMENT).
Например, вот так: `/choose AAPL 10000`
После этого бот автоматически загрузит исторические данные о стоимости акций (за последние 2 года), обучит несколько моделей временных рядов, выберет наилучшую по метрикам качества и построит прогноз на ближайшие 30 дней.
Тут придется немного подождать, пока нейронная сеть будет обучаться и создавать прогноз.
В результате ты получишь прогноз стоимости акций и рекомендации на каждый день.

**Доступные команды:**
`/choose TICKER INVESTMENT` - анализ акций и формирование прогноза
Пример: `/choose AAPL 10000`

`/start` или `/help` - получение справки

`/test TICKER` - тестирование загрузки данных по тикеру

`/exit` - выйти из бота

**Особенности:**
• Анализ только по рабочим дням (Пн-Пт)
• Прогноз на 30 рабочих дней (~6 недель)
• Три модели: Random Forest, ARIMA, LSTM
• Автоматический выбор лучшей модели

**Примеры тикеров:**
AAPL (Apple), TSLA (Tesla), GOOGL (Google), MSFT (Microsoft)


    """
    bot.send_message(message.chat.id, welcome_text, parse_mode='Markdown')

@bot.message_handler(commands=['test'])
def handle_test(message):
    """Тестовая команда для проверки данных"""
    try:
        company = message.text.split()[1] if len(message.text.split()) > 1 else "AAPL"
        company = company.upper()
        
        bot.send_message(message.chat.id, f"🔍 Тестирую загрузку данных для {company}...")
        
        data = load_data(company)
        
        result = f"""
📊 **Тест данных для {company}:**

• Загружено записей: {len(data)}
• Рабочих дней: {len(data)} 
• Первая дата: {data.index[0].strftime('%Y-%m-%d')}
• Последняя дата: {data.index[-1].strftime('%Y-%m-%d')}
• Последняя цена: ${float(data.iloc[-1]):.2f}

✅ Данные загружены успешно!
        """
        bot.send_message(message.chat.id, result, parse_mode='Markdown')
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка: {str(e)}")

@bot.message_handler(commands=['exit'])
def handle_exit(message):
    """Обработчик команды выхода из бота"""
    try:
        user_id = message.from_user.id
        user_name = message.from_user.first_name
        
        goodbye_text = f"""
👋 **До свидания, {user_name}!**

Спасибо за использование бота для анализа акций!

Если захотите снова проанализировать акции, просто напишите `/start`

📊 *Удачных инвестиций!*
        """
        bot.send_message(message.chat.id, goodbye_text, parse_mode='Markdown')
        
        # Логирование выхода
        logging.info(f"UserID: {user_id} - вышел из бота")
        
    except Exception as e:
        bot.send_message(message.chat.id, "👋 До свидания!")

if __name__ == "__main__":
    print("🚀 Бот запущен...")
    bot.polling()