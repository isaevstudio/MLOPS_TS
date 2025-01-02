# How the App Works
This project focuses on stock price prediction. The steps outlined in the main README file will prepare and split the data by days, 
simulating upcoming days that need to be predicted.

The application introduces a 3-second delay, which we treat as one day (3 seconds = 1 day). 
The prediction occurs for the next day, and when you open the localhost, the StreamingResponse will continuously make predictions and display them in the browser every 3 seconds,
with no need to refresh the web page.


# Info on trained model

The model is trained to predict 10 days ahead, but for simplicity, 
we'll predict only 1 day ahead and ignore the remaining 9 days.

For future work, the approach will be:

- Predict day 1 and the next 10 days.
- Predict day 2, update predictions from day 2 to day 10, and add day 11.


The reason for updating is that predictions 10 days ahead from day 1 are less accurate 
compared to those from day 2. Therefore, the predictions for day 2 will be more reliable than those for day 1.

As we move to day 3, the 10-day predictions will become even more accurate
than those for day 1 and 2, and so on.
