from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_CSV_PATH = Path(r"C:\Users\asus\Downloads\credit card\creditcard.csv")
FEATURE_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data(csv_source):
    data = pd.read_csv(csv_source)
    missing_columns = set(FEATURE_COLUMNS + ["Class"]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = data[FEATURE_COLUMNS + ["Class"]].dropna()
    data["Class"] = data["Class"].astype(int)
    return data


@st.cache_resource(show_spinner=False)
def train_model(data):
    legit = data[data["Class"] == 0]
    fraud = data[data["Class"] == 1]

    if fraud.empty:
        raise ValueError("No fraud rows found. The Class column must contain value 1.")
    if legit.empty:
        raise ValueError("No legitimate rows found. The Class column must contain value 0.")

    sample_size = min(len(legit), len(fraud))
    legit_sample = legit.sample(n=sample_size, random_state=2)
    fraud_sample = fraud.sample(n=sample_size, random_state=2)
    balanced_data = pd.concat([legit_sample, fraud_sample], axis=0)

    x = balanced_data[FEATURE_COLUMNS]
    y = balanced_data["Class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=2,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=2)),
        ]
    )
    model.fit(x_train, y_train)

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    metrics = {
        "balanced_rows": len(balanced_data),
        "train_accuracy": accuracy_score(y_train, train_predictions),
        "test_accuracy": accuracy_score(y_test, test_predictions),
        "confusion_matrix": confusion_matrix(y_test, test_predictions),
        "classification_report": classification_report(
            y_test,
            test_predictions,
            target_names=["Legitimate", "Fraud"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return model, metrics


def prediction_label(value):
    return "Fraud" if int(value) == 1 else "Legitimate"


def prediction_status(actual, predicted):
    if int(actual) == int(predicted):
        return "Correct"
    return "Wrong"


def predict_one(model, row):
    features = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(features)[0])
    probability = model.predict_proba(features)[0]
    return prediction, float(probability[1])


st.title("Credit Card Fraud Detection")
st.caption("Logistic Regression model using the uploaded creditcard.csv dataset")

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])
    use_default = st.checkbox(
        "Use CSV from Downloads",
        value=DEFAULT_CSV_PATH.exists(),
        disabled=uploaded_file is not None,
    )
    st.code(str(DEFAULT_CSV_PATH), language="text")

    st.header("Labels")
    st.write("0 = Legitimate transaction")
    st.write("1 = Fraud transaction")

csv_source = uploaded_file if uploaded_file is not None else DEFAULT_CSV_PATH

try:
    if uploaded_file is None and not use_default:
        st.info("Upload a CSV file or enable the default CSV path in the sidebar.")
        st.stop()

    with st.spinner("Loading data and training model..."):
        df = load_data(csv_source)
        model, metrics = train_model(df)
except Exception as exc:
    st.error(f"Could not start the app: {exc}")
    st.stop()

total_rows = len(df)
class_counts = df["Class"].value_counts().rename(index={0: "Legitimate", 1: "Fraud"})

overview_tab, reviewer_tab, manual_tab, validation_tab = st.tabs(
    ["Overview", "Reviewer Check", "Manual Prediction", "Validation"]
)

with overview_tab:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total rows", f"{total_rows:,}")
    col2.metric("Legitimate rows", f"{int((df['Class'] == 0).sum()):,}")
    col3.metric("Fraud rows", f"{int((df['Class'] == 1).sum()):,}")
    col4.metric("Balanced training rows", f"{metrics['balanced_rows']:,}")

    st.subheader("Class Distribution")
    st.bar_chart(class_counts)

    st.subheader("Sample Data")
    st.dataframe(df.head(20), width="stretch")

with reviewer_tab:
    st.subheader("Show True vs Predicted")
    st.write(
        "Choose real transactions from the CSV. The table shows the actual class, "
        "the model prediction, and whether the model is correct or wrong."
    )

    col1, col2 = st.columns([1, 1])
    selected_class = col1.selectbox(
        "Rows to review",
        options=["Mixed", "Only Legitimate", "Only Fraud"],
    )
    sample_count = col2.slider("Number of rows", min_value=5, max_value=50, value=10, step=5)

    if selected_class == "Only Legitimate":
        review_df = df[df["Class"] == 0].sample(n=sample_count, random_state=10)
    elif selected_class == "Only Fraud":
        available = min(sample_count, int((df["Class"] == 1).sum()))
        review_df = df[df["Class"] == 1].sample(n=available, random_state=10)
    else:
        review_df = df.sample(n=sample_count, random_state=10)

    result_rows = []
    for original_index, row in review_df.iterrows():
        actual = int(row["Class"])
        predicted, fraud_probability = predict_one(model, row[FEATURE_COLUMNS])
        result_rows.append(
            {
                "CSV row": original_index,
                "Amount": float(row["Amount"]),
                "Actual value": actual,
                "Actual label": prediction_label(actual),
                "Predicted value": predicted,
                "Predicted label": prediction_label(predicted),
                "Fraud probability": round(fraud_probability, 4),
                "Result": prediction_status(actual, predicted),
            }
        )

    result_df = pd.DataFrame(result_rows)
    st.dataframe(result_df, width="stretch", hide_index=True)

    correct_count = int((result_df["Result"] == "Correct").sum())
    wrong_count = int((result_df["Result"] == "Wrong").sum())
    col1, col2 = st.columns(2)
    col1.metric("Correct predictions", correct_count)
    col2.metric("Wrong predictions", wrong_count)

with manual_tab:
    st.subheader("Predict One Transaction")
    st.write("Paste 30 comma-separated feature values in this order:")
    st.code(", ".join(FEATURE_COLUMNS), language="text")

    example = df[FEATURE_COLUMNS].iloc[0]
    st.caption("Example input from the first row:")
    st.code(",".join(str(value) for value in example.values), language="text")

    user_input = st.text_area("Transaction features", height=120)
    if st.button("Predict Transaction"):
        try:
            values = [float(item.strip()) for item in user_input.replace("\n", ",").split(",") if item.strip()]
            if len(values) != len(FEATURE_COLUMNS):
                st.error(f"Expected {len(FEATURE_COLUMNS)} values, but got {len(values)}.")
            else:
                row = pd.Series(values, index=FEATURE_COLUMNS)
                prediction, fraud_probability = predict_one(model, row)
                if prediction == 1:
                    st.error(f"Prediction: Fraud transaction. Fraud probability: {fraud_probability:.2%}")
                else:
                    st.success(
                        f"Prediction: Legitimate transaction. Fraud probability: {fraud_probability:.2%}"
                    )
        except ValueError:
            st.error("Please enter only numeric values separated by commas.")

with validation_tab:
    st.subheader("Model Validation")
    col1, col2 = st.columns(2)
    col1.metric("Training accuracy", f"{metrics['train_accuracy']:.2%}")
    col2.metric("Testing accuracy", f"{metrics['test_accuracy']:.2%}")

    cm = metrics["confusion_matrix"]
    st.write("Confusion matrix on the balanced test split")
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual Legitimate", "Actual Fraud"],
            columns=["Predicted Legitimate", "Predicted Fraud"],
        ),
        width="stretch",
    )

    report = pd.DataFrame(metrics["classification_report"]).transpose()
    st.write("Precision, recall, and F1 score")
    st.dataframe(report, width="stretch")

    st.subheader("What changed from the notebook")
    st.markdown(
        """
- Fixed the CSV path so it points to your provided `creditcard.csv`.
- Used only legitimate rows when undersampling normal transactions.
- Wrapped scaling and logistic regression in one pipeline so prediction uses the same preprocessing as training.
- Added reviewer-friendly screens that compare actual labels with predicted labels.
- Added validation metrics so you can explain model quality instead of only showing one prediction.
"""
    )
