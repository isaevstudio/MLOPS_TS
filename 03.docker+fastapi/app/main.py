import warnings
import time
import os
import nltk
# nltk.download('all')
from dotenv import load_dotenv

# importing the functions from the modules
from app.preprocessing.prepare_data import df_date_convert, from_csv_summarize, from_csv_sentiment, start_inte
from app.model.train import model_training
from app.preprocessing.prepare_data import scale_data, create_sequences
from app.utils.utils import metrics, plot_TSMixer_results
from app.dto.dto import AppConfig

#importing the libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers import BertForSequenceClassification, BertTokenizer

# import fastapi
from fastapi import FastAPI
from typing import Dict
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
import json

# indicating the device to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


warnings.filterwarnings('ignore')

load_dotenv()
scaler = MinMaxScaler()

app = FastAPI()
config = AppConfig()


# params
RANDOM_STATE = 42
train_size_ratio = 0.8
delay = 3 # in seconds; 3 seconds equavalent to 1 day.

'''
The model is trained to predict 10 days ahead, but for simplicity, 
we'll predict only 1 day ahead and ignore the remaining 9 days.

For future work, the approach will be:

- Predict day 1 and the next 10 days.
- Predict day 2, update predictions from day 2 to day 10, and add day 11.


The reason for updating is that predictions 10 days ahead from day 1 are less accurate 
compared to those from day 2. Therefore, the predictions for day 2 will be more reliable than those for day 1.

As we move to day 3, the 10-day predictions will become even more accurate
than those for day 1 and 2, and so on.

'''


# Load the fine-tuned sentiment model and tokenizer
model_sentiment = BertForSequenceClassification.from_pretrained(config.finetuned_model_path)
tokenizer_sentiment = BertTokenizer.from_pretrained(config.finetuned_model_path)


# Paths for other operations
df_predict_path = config.df_predict_path # Can be deleted. This on is covered in previous app
future_imitate = config.future_imitate # The path to the data where the raw price and news are merged and fed to the model

news_by_day = config.news_by_day # raw data for imitatting the stock price
prices_by_day = config.prices_by_day # raw data for imitatting the stock price

# Easiest way to ensure that the validation is working, dto has a function that incorporates the model name to the path
model_file = config.model_file 




@app.get("/readiness")
async def helth_check() -> Dict:
    return {"status": "success"}

'''
@app.get("/preparePredict")
async def get_predict() -> Dict:

# ------------------------------------ Data predict + hypertune
  
    df_predict = pd.read_csv(df_predict_path)

    # move to data prep
    df_predict = scale_data(df_predict)
    df_predict[['close', 'volume']] = scaler.fit_transform(df_predict[['close', 'volume']])

    values = df_predict.drop('date', axis=1).values
# ------------------------------------ Prediction
    seq_length = 180 # Number of timesteps
    lr = 0.001
    epochs=50
    hidden_size = 64
    num_layers = 2
    predict_horizon = 10

    model_path = os.path.join(saved_model, 'm_TSMixer.pkl')


    X, y = create_sequences(values, seq_length, predict_horizon)
    train_size = int(X.shape[0] * train_size_ratio)

    print(train_size)

    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    try:
        print("Trying to load the model...")
        with open(saved_model+'m_TSMixer.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")

    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            print(f"Model loading failed: {e}")
            print(f"Training the model!")

            os.makedirs(saved_model, exist_ok=True) # creating a directory to save the model

            # Train the model
            model = model_training(values, X_train.shape[2], train_loader, lr, epochs, hidden_size, num_layers, output_size=predict_horizon)

            # Save the trained model
            with open(saved_model+'m_TSMixer.pkl', 'wb') as f:
                pickle.dump(model, f)

            print("Model trained and saved successfully.")

    # Use the loaded model
   
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            pred = model(batch_X)
            predictions.append(pred)

    # Convert predictions to a numpy array and invert normalization
    # Create a placeholder for inverse scaling
    predictions = torch.cat(predictions).numpy()

    # Initialize an array of zeros with the same number of features as the original data
    predicted_prices_full = np.zeros((predictions.shape[0], df_predict[['close', 'volume']].shape[1]))

    # Place the predictions in the 'Close' column (3rd index)
    predicted_prices_full[:, 0] = predictions[:, 0]

    # Inverse transform
    predicted_prices_full = scaler.inverse_transform(predicted_prices_full)

    # Extract the 'Close' prices from the inverse-transformed data
    predicted_prices = predicted_prices_full[:, 0]

    # Convert predictions to a DataFrame and save to a CSV file
    predicted_prices_df = pd.DataFrame(predicted_prices, columns=['predicted_close'])


# ---------------------------------------------- Evaluation

    y_test_np = y_test.numpy().reshape(-1, 1)

    # Create an array with the same shape as the original features, filled with zeros
    y_test_full = np.zeros((y_test_np.shape[0], df_predict[['close', 'volume']].shape[1]))

    # Place y_test_np in the Close column (assuming it's the 4th column as before)
    y_test_full[:, 0] = y_test_np[:, 0]

    # Apply inverse transform only on the relevant column
    actual_prices_full = scaler.inverse_transform(y_test_full)

    # Extract the actual Close prices
    actual_prices = actual_prices_full[:, 0]

    # Evaluate the model
    results_eval, mae, mse, r2, mape = metrics(test=actual_prices, prediction=predicted_prices)
    print(results_eval)


    to_save = predicted_prices_df[['date', 'Actual_Close']]
    to_save.to_csv('../dataset_new/plots/org_plot.csv')

    # Visualization
    fig = plot_TSMixer_results(predicted_prices_df=predicted_prices_df, date='date', actual='Actual_Close', predicted='Predicted_Close')
    print(fig)

    return {"r2":round(r2,2), "mae":round(mae,2), "mse":round(mse,2), "mape":round(mape,2),}
'''


def predict():

    fresh_preds=pd.DataFrame(columns=['date', 'pred_close'])

    for file in os.listdir(prices_by_day):

            temp_price=pd.read_csv(os.path.join(prices_by_day, file))

            symbol=str(temp_price['Date'].values[0])

            try:
                temp_news=pd.read_csv(news_by_day+file)

                temp_news = from_csv_summarize(temp_news, symbol)
                temp_news = from_csv_sentiment(temp_news, model_sentiment, tokenizer_sentiment, device)

            except:
                data={
                    'Date':temp_price['Date'],
                    'Text':np.nan,
                    'Url':np.nan,
                    'Mark':np.nan   
                }
                temp_news=pd.DataFrame(data)

                temp_news['sentiment'] = 0 # neutral sentiment
                temp_news.columns=temp_news.columns.str.lower()


            temp_news.columns=temp_news.columns.str.lower()
            temp_price.columns=temp_price.columns.str.lower()

            try:
                start_inte(temp_news, temp_price, symbol, saving_path=future_imitate)
            except:
                os.makedirs(future_imitate, exist_ok=True)
                
                start_inte(temp_news, temp_price, symbol, saving_path=future_imitate)

    
            df_predict = pd.read_csv(os.path.join(future_imitate, f"{symbol}.csv"))


            # Scaling
            df_predict = scale_data(df_predict)
            df_predict[['close', 'volume']] = scaler.fit_transform(df_predict[['close', 'volume']])
            try:
                df_predict['scaled_sentiment']
            except:
                df_predict['scaled_sentiment'] = 0

            values = df_predict.drop('date', axis=1).values

            X_predictions_loader = DataLoader(TensorDataset(torch.tensor(values, dtype=torch.float32)), batch_size=32, shuffle=False)
            

            # Load the model
            with open(model_file, 'rb') as f:
                model = pickle.load(f)


            model.eval()
            
            predictions = []
            with torch.no_grad():
                for batch_X in X_predictions_loader:
                    pred = model(batch_X[0])
                    predictions.append(pred)

            # Convert predictions to a numpy array and invert normalization
            # Create a placeholder for inverse scaling
            predictions = torch.cat(predictions).numpy()

            # Initialize an array of zeros with the same number of features as the original data
            predicted_prices_full = np.zeros((predictions.shape[0], df_predict[['close', 'volume']].shape[1]))

            # Place the predictions in the 'Close' column (3rd index)
            predicted_prices_full[:, 0] = predictions[:, 0]

            # Inverse transform
            predicted_prices_full = scaler.inverse_transform(predicted_prices_full)

            # Extract the 'Close' prices from the inverse-transformed data
            predicted_prices = predicted_prices_full[:, 0]

            # Convert predictions to a DataFrame and save to a CSV file
            predicted_prices_df = pd.DataFrame(predicted_prices, columns=['predicted_close'])


            df_predict['date']=pd.to_datetime(df_predict['date']).dt.date

            # Combine dates, actual prices, and predicted prices into a DataFrame
            predicted_prices_df = pd.DataFrame({
                'date': pd.to_datetime(df_predict['date'].values[0]),  # Ensure the date is in correct format
                'pred_close': predicted_prices
            })

            # Concatenate the fresh predictions to the existing DataFrame
            fresh_preds = pd.concat([fresh_preds, predicted_prices_df], ignore_index=True)

            # Convert the 'date' column to ISO format using .dt accessor and then to dictionary
            predicted_prices_df['date'] = predicted_prices_df['date'].dt.strftime('%Y-%m-%dT%H:%M:%S')  # ISO format string

            # Convert to dictionary with records
            prediction_json = predicted_prices_df.to_dict(orient="records")

            # Yield the prediction data as part of the event stream
            yield f"data: {json.dumps(prediction_json)}\n\n"

            # Sleep before sending the next batch of predictions
            time.sleep(delay)

@app.get("/stream_predictions")
async def stream_predictions():
    return StreamingResponse(predict(), media_type="text/event-stream")


