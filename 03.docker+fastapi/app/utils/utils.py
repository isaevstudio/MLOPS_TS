import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error



def metrics(test, prediction)->pd.DataFrame:

    print('def metrics')

    mae = mean_absolute_error(test, prediction)
    mse = mean_squared_error(test, prediction)
    r2 = r2_score(test, prediction)
    mape = mean_absolute_percentage_error(test, prediction) * 100

    data = {
            'eval':['MAE','MSE','R2', 'MAPE'], 
            'score':[round(mae, 2), round(mse, 2), round(r2, 2), round(mape, 2)]
            }
    
    results = pd.DataFrame(data)
    return results, mae, mse, r2, mape


"""
def metrics(test: np.ndarray, prediction: np.ndarray, rounding: int = 2) -> pd.DataFrame:
 
    if test.shape[0] != prediction.shape[0]:
        raise ValueError("Mismatch in number of samples between test and prediction.")

    if len(test.shape) == 1:  # If test is 1D, reshape it for multi-step comparison
        test = test.reshape(prediction.shape)

    metrics_per_day = []
    for day in range(prediction.shape[1]):
        day_test = test[:, day]
        day_prediction = prediction[:, day]
        mae = mean_absolute_error(day_test, day_prediction)
        mse = mean_squared_error(day_test, day_prediction)
        r2 = r2_score(day_test, day_prediction)
        mape = mean_absolute_percentage_error(day_test, day_prediction) * 100
        metrics_per_day.append([day + 1, mae, mse, r2, mape])

    # Create a DataFrame with results
    metrics_df = pd.DataFrame(metrics_per_day, columns=['Day', 'MAE', 'MSE', 'R2', 'MAPE'])
    return metrics_df
"""

def plot_TSMixer_results(predicted_prices_df, date, actual, predicted):
   
    print('plot TSMixer')

    plt.figure(figsize=(12, 6))
    plt.plot(predicted_prices_df[date], predicted_prices_df[actual], label='Actual Close Price')
    plt.plot(predicted_prices_df[date], predicted_prices_df[predicted], label='Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Prices')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()