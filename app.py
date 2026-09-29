import gradio as gr
import pandas as pd
import joblib


# =========================
# Load Saved Model Files
# =========================

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
metadata = joblib.load("metadata.pkl")

feature_columns = metadata["feature_columns"]
categorical_columns = metadata["categorical_columns"]
choices = metadata["choices"]


# =========================
# Prediction Function
# =========================

def predict_dropout(*values):

    # Create one-row DataFrame
    input_data = pd.DataFrame(
        [values],
        columns=feature_columns
    )

    # Convert all values to numeric
    input_data = input_data.astype(float)

    # Scale input using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get probabilities
    probabilities = model.predict_proba(input_scaled)[0]

    dropout_probability = probabilities[1]

    # Project-defined risk categories
    if dropout_probability < 0.30:
        risk = "Low Risk"
    elif dropout_probability < 0.60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    prediction_label = (
        "Dropout"
        if prediction == 1
        else "Not Dropout"
    )

    return (
        prediction_label,
        f"{dropout_probability * 100:.2f}%",
        risk
    )


# =========================
# Create Input Components
# =========================

inputs = []

for column in feature_columns:

    if column in categorical_columns:

        component = gr.Dropdown(
            choices=choices[column],
            label=column,
            value=choices[column][0]
        )

    else:

        component = gr.Number(
            label=column
        )

    inputs.append(component)


# =========================
# Gradio Interface
# =========================

with gr.Blocks(title="Student Dropout Prediction") as demo:

    gr.Markdown(
        """
        # Student Dropout Prediction

        Enter student information to predict dropout status,
        dropout probability, and risk category.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Student Information")

            for i in range(0, 18):
                inputs[i].render()

        with gr.Column():
            gr.Markdown("### Academic & Economic Information")

            for i in range(18, len(inputs)):
                inputs[i].render()

    predict_button = gr.Button("Predict Dropout Risk")

    prediction_output = gr.Textbox(
        label="Prediction"
    )

    probability_output = gr.Textbox(
        label="Dropout Probability"
    )

    risk_output = gr.Textbox(
        label="Risk Category"
    )

    predict_button.click(
        fn=predict_dropout,
        inputs=inputs,
        outputs=[
            prediction_output,
            probability_output,
            risk_output
        ]
    )


# =========================
# Launch Application
# =========================
import os

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
