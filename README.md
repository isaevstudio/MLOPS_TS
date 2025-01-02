# Objective (This section will be updated as needed based on the latest version of the app uploaded):
- integrate fastapi and pydantic
- dockerize the app
where, pydantic is used for path validation in this project. Without Pydantic, a path error would occur when triggering the endpoint. By leveraging Pydantic, we can ensure that all paths are validated before running the project, preventing errors during execution.

# Steps to Run the Project
1. Download Necessary Files
Follow the instructions to download all the required files from Google Drive.
2. Prepare Raw Data
Execute the dataset_prepare_raw_data.ipynb notebook to generate the raw data needed for predictions.
3. Set Up Environment Variables
Create a .env file to configure the paths required by the project. Use the following structure and the paths:
    df_predict_path='/app/dataset_new/price_news_integrate/'
    finetuned_model_path='/app/finetuned_finbert_model/'
    model_file='/app/saved_model/'
    integrated_data_path='/app/dataset_new/price_news_integrate/'
    saved_model='/app/saved_model/'
    future_imitate='/app/dataset_new/future_imitate/'
    prices_by_day='/app/dataset_new/prices_by_day/'
    news_by_day='/app/dataset_new/news_by_day/'
   
4. Download Missing Files and Models
After downloading the missing files and generating the raw data, proceed to run the model.py file.


# Important Notes
Due to size limitations, the fine-tuned FinBERT model and the saved TSMIXER model are not included in the repository.
Download the missing folders from this link: https://drive.google.com/drive/folders/1Di09f_7f2wqDtsTu_p0yDpTDQf8H7lT1?usp=sharing/


## Missing folders:
1. saved_model
2. finetuned_finbert_model

## Schema (The dataset_new folder at least should contain the stock_price_data_raw & news_data_raw):

![Screenshot 2025-01-03 at 5 46 20 AM](https://github.com/user-attachments/assets/3a8ba399-f75c-43d1-8348-5e7a1c7b499c)

__ P.S.__ For simplicity, the project has been simplified to its maximum extent.
