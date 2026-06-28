# Credit Card Fraud Detection App

This project is a Streamlit web application for detecting fraudulent credit card transactions using machine learning models. The app allows users to upload credit card transaction data, preprocess it, and detect anomalies that indicate potential fraud.

## Features

- **Data Upload:** Users can upload their own credit card transaction datasets in CSV format.
- **Data Preprocessing:** The app standardizes the transaction amounts and time features, and applies Principal Component Analysis (PCA) to reduce dimensionality.
- **Anomaly Detection:** Uses machine learning models like Isolation Forest to detect fraudulent transactions.
- **Visualization:** Provides scatter plots to visualize anomalies within PCA-transformed components.
- **Interactive Interface:** Built with Streamlit, allowing easy interaction and visualization of results.

## How to Use

1. **Upload Data:** Click on "Choose a file" to upload your transaction data.
2. **View Results:** The app will display the processed data, highlighting fraudulent transactions.
3. **Visualization:** Use the built-in visualizations to explore the detected anomalies.

## Deployment

The app is deployed on Streamlit Community Cloud and can be accessed [here](https://creditcardfrauddetection-8huvkevjzeddgqevvrno74.streamlit.app/).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or new features.

## Acknowledgments

- **Data Source:** The credit card transaction dataset used in this project is obtained from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- **Machine Learning Models:** The app leverages models provided by the `scikit-learn` library.

## Contact

If you have any questions or feedback, feel free to reach out via GitHub or email.
