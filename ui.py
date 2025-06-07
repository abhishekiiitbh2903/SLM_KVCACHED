import streamlit as st
import requests
from typing import List

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="SLM Playground", layout="wide")

st.markdown("## 🔍 SLM Model Inference Playground")

col_input, col_output = st.columns([1, 2])

with col_input:
    st.markdown("### ✏️ Input & Settings")

    input_text = st.text_area("Input Text", placeholder="Type your sentence here...", height=150)

    model_options = ["untrained", "basic", "cached"]
    selected_models: List[str] = st.multiselect(
        "Select Model(s)", model_options, default=["basic"]
    )

    run_button = st.button("🚀 Run")

    results = {}

    if run_button and input_text.strip():
        if not selected_models:
            st.warning("Please select at least one model.")
        else:
            for model_type in selected_models:
                with st.spinner(f"Running {model_type.capitalize()} model..."):
                    try:
                        response = requests.post(API_URL, json={
                            "input_text": input_text,
                            "model_type": model_type
                        }, timeout=30)

                        if response.status_code == 200:
                            result = response.json()
                            result_text = result["result"]
                            time_taken = result["time"]
                        else:
                            result_text = f"Error: {response.text}"
                            time_taken = 0.0
                    except Exception as e:
                        result_text = f"Exception: {e}"
                        time_taken = 0.0

                    results[model_type] = {
                        "result": result_text,
                        "time": time_taken
                    }

    elif run_button and not input_text.strip():
        st.warning("Please enter input text.")

with col_output:
    st.markdown("### 🧠 Model Output")

    if run_button:
        if results:
            for model_type, output in results.items():
                st.markdown(f"**{model_type.capitalize()} Model**")
                st.text_area(
                    label="Output",
                    value=output["result"],
                    height=150,
                    disabled=True,
                    key=f"output_{model_type}"
                )
                st.success(f"⏱️ Time Taken: {output['time']:.3f} seconds")
                st.markdown("---")
        else:
            st.info("No output available.")
