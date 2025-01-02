# The first two sections, explaining how the app works and the prediction for 10 days ahead, can be skipped. Starting from section 3 MUST read.

# 1. How the App Works
This project focuses on stock price prediction. The steps outlined in the main README file will prepare and split the data by days, 
simulating upcoming days that need to be predicted.

The application introduces a 3-second delay, which we treat as one day (3 seconds = 1 day). 
The prediction occurs for the next day, and when you open the localhost, the StreamingResponse will continuously make predictions and display them in the browser every 3 seconds,
with no need to refresh the web page.


# 2. Info on trained model

The model is trained to predict 10 days ahead, but for simplicity, 
we'll predict only 1 day ahead and ignore the remaining 9 days.

For future work, the approach will be:

- Predict day 1 and the next 10 days.
- Predict day 2, update predictions from day 2 to day 10, and add day 11.


The reason for updating is that predictions 10 days ahead from day 1 are less accurate 
compared to those from day 2. Therefore, the predictions for day 2 will be more reliable than those for day 1.

As we move to day 3, the 10-day predictions will become even more accurate
than those for day 1 and 2, and so on.


# 3. Objective:
- integrate fastapi and pydantic
- dockerize the app
where, pydantic is used for path validation in this project. Without Pydantic, a path error would occur when triggering the endpoint. By leveraging Pydantic, we can ensure that all paths are validated before running the project, preventing errors during execution.

# 4. Steps to Run the Project
1. Download Necessary Files
Follow the instructions to download all the required files from Google Drive.
2. Prepare Raw Data
Execute the dataset_prepare_raw_data.ipynb notebook to generate the raw data needed for predictions.
3. Set Up Environment Variables
Create a .env file to configure the paths required by the project. Use the following structure and the paths: <br>
    - df_predict_path='/app/dataset_new/price_news_integrate/' <br>
    - finetuned_model_path='/app/finetuned_finbert_model/' <br>
    - model_file='/app/saved_model/' <br>
    - integrated_data_path='/app/dataset_new/price_news_integrate/' <br>
    - saved_model='/app/saved_model/' <br>
    - future_imitate='/app/dataset_new/future_imitate/' <br>
    - prices_by_day='/app/dataset_new/prices_by_day/' <br>
    - news_by_day='/app/dataset_new/news_by_day/' <br>
   
4. Download Missing Files and Models
After downloading the missing files and generating the raw data, proceed to run the model.py file.


# 5. Important Notes
Due to size limitations, the fine-tuned FinBERT model and the saved TSMIXER model are not included in the repository.
Download the missing folders from this link: https://drive.google.com/drive/folders/1Di09f_7f2wqDtsTu_p0yDpTDQf8H7lT1?usp=sharing/


## Missing folders:
1. saved_model
2. finetuned_finbert_model

## Schema (The dataset_new folder at least should contain the stock_price_data_raw & news_data_raw):

![Screenshot 2025-01-03 at 5 46 20 AM](https://github.com/user-attachments/assets/3a8ba399-f75c-43d1-8348-5e7a1c7b499c)

__ P.S.__ For simplicity, the project has been simplified to its maximum extent.
