import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any

from src.data_manager import DataManager
from src.model_factory import ModelFactory
from src.evaluator import Evaluator

# Set page configuration first
st.set_page_config(page_title="AutoML Benchmark", layout="wide", page_icon="🏅")

def render_sidebar() -> Dict[str, Any]:
    """
    Renders the sidebar interface and returns user inputs.
    """
    st.sidebar.header("1. Data Selection")
    
    # 1. Dataset Selection
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
    dataset_option = "iris" # Default
    if not uploaded_file:
        dataset_option = st.sidebar.selectbox(
            "Or choose a sample dataset", 
            ["iris", "wine", "breast_cancer", "digits", "titanic", "penguins"]
        )

    st.sidebar.markdown("---")
    st.sidebar.header("2. Model Selection")
    
    # 2. Model Selection
    model_factory = ModelFactory()
    available_models = model_factory.get_models()
    selected_model_names = st.sidebar.multiselect(
        "Select Models to Train", 
        list(available_models.keys()), 
        default=list(available_models.keys())
    )
    
    run_button = st.sidebar.button("Run Benchmark", type="primary")
    
    return {
        "uploaded_file": uploaded_file,
        "dataset_option": dataset_option,
        "selected_model_names": selected_model_names,
        "available_models": available_models,
        "run_button": run_button
    }

def main():
    """
    Main execution entry point for the Streamlit dashboard.
    """
    st.title("🏅 ML Model Comparison Dashboard")
    st.markdown("### 🤖 Automated Benchmarking for Classifiers")
    st.markdown("Upload your dataset (CSV) or use a sample one to train and compare 5 distinct ML models instantly.")

    # Get inputs from sidebar
    inputs = render_sidebar()

    if inputs["run_button"]:
        # === Step 1: Data Loading ===
        dm = DataManager()
        
        try:
            if inputs["uploaded_file"] is not None:
                df = dm.load_data(file=inputs["uploaded_file"])
                st.info("✅ Using uploaded dataset.")
            else:
                df = dm.load_data(sample_dataset=inputs["dataset_option"])
                st.info(f"✅ Using sample dataset: {inputs['dataset_option']}")
            
            st.subheader("📊 Dataset Preview")
            st.dataframe(df.head())
            st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

            # === Step 2: Configuration ===
            # Allow user to select target if using custom data
            target_col = df.columns[-1] 
            if inputs["uploaded_file"] is not None:
                cols = list(df.columns)
                target_col = st.selectbox("Select Target Column (Y)", cols, index=len(cols)-1)
            else:
                st.markdown(f"**Target Column:** `{target_col}`")
            
            # === Step 3: Preprocessing ===
            with st.status("Preprocessing data..."):
                X_train, X_test, y_train, y_test = dm.preprocess_data(df, target_column=target_col)
                st.write("Data loaded, cleaned, and split into Train/Test sets.")
                st.write("Features scaled using StandardScaler.")

            # Filter selected models
            models_to_train = {
                name: inputs["available_models"][name] 
                for name in inputs["selected_model_names"]
            }
            
            if not models_to_train:
                st.warning("⚠️ Please select at least one model to train.")
                return

            # === Step 4: Training & Evaluation ===
            evaluator = Evaluator()
            with st.spinner(f"Training {len(models_to_train)} models..."):
                results_df = evaluator.train_and_evaluate(models_to_train, X_train, X_test, y_train, y_test)
            
            st.success("🎉 Benchmarking Complete!")
            
            # === Step 5: Visualization ===
            st.divider()
            st.subheader("🏆 Model Leaderboard")
            st.dataframe(
                results_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1 Score"], color='#dbf2da'),
                use_container_width=True
            )
            
            st.subheader("📈 Performance Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Accuracy Comparison**")
                fig, ax = plt.subplots()
                sns.barplot(x="Accuracy", y="Model", data=results_df, ax=ax, palette="viridis")
                ax.set_xlim(0, 1.0)
                st.pyplot(fig)
                
            with col2:
                st.markdown("**All Metrics Details**")
                melted_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
                fig, ax = plt.subplots()
                sns.barplot(x="Score", y="Model", hue="Metric", data=melted_df, ax=ax, palette="rocket")
                ax.set_xlim(0, 1.0)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
