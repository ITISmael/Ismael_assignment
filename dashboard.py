import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Load model (placeholder - replace with your actual model loading code)
# model = pickle.load(open('selected_features_with_target.pkl', 'rb'))
# For now, we'll create mock predictions

# Dashboard title
st.title("📊 Student Performance Prediction Dashboard")

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Model Input Parameters")
    
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    assignment_score = st.slider("Assignment Score", 0, 100, 70)
    midterm_score = st.slider("Midterm Score", 0, 100, 65)
    final_score = st.slider("Final Score", 0, 100, 60)
    outstanding_balance = st.number_input("Outstanding Balance ($)", 0, 10000, 500)
    
    # Mock prediction function (replace with actual model prediction)
    def predict_performance(attendance, assignment, midterm, final, balance):
        # This is a placeholder - replace with your actual model prediction
        weights = np.array([0.2, 0.25, 0.25, 0.25, -0.0001])
        inputs = np.array([attendance, assignment, midterm, final, balance])
        raw_score = np.dot(weights, inputs)
        probability = 1 / (1 + np.exp(-0.1 * (raw_score - 50)))  # Sigmoid function
        return probability
    
    if st.button("Predict Performance"):
        prediction = predict_performance(attendance, assignment_score, midterm_score, 
                                       final_score, outstanding_balance)
        st.session_state['prediction'] = prediction
        st.session_state['input_data'] = {
            'Attendance': attendance,
            'Assignment_Score': assignment_score,
            'Midterm_Score': midterm_score,
            'Final_Score': final_score,
            'Outstanding_Balance': outstanding_balance
        }

# Main dashboard layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Feature Analysis")
    
    # Feature importance visualization (mock data)
    feature_importance = pd.DataFrame({
        'Feature': ['Attendance', 'Assignment_Score', 'Midterm_Score', 'Final_Score', 'Outstanding_Balance'],
        'Importance': [0.20, 0.25, 0.25, 0.25, -0.05]  # Mock importance values
    })
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
    plt.title("Feature Importance in Model")
    st.pyplot(fig)
    
    # Correlation heatmap (mock data)
    st.subheader("🔍 Feature Correlations")
    mock_data = pd.DataFrame({
        'Attendance': np.random.normal(attendance, 10, 100),
        'Assignment_Score': np.random.normal(assignment_score, 10, 100),
        'Midterm_Score': np.random.normal(midterm_score, 10, 100),
        'Final_Score': np.random.normal(final_score, 10, 100),
        'Outstanding_Balance': np.random.normal(outstanding_balance, 200, 100)
    })
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(mock_data.corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

with col2:
    st.subheader("🔮 Performance Prediction")
    
    if 'prediction' in st.session_state:
        # Display prediction result
        prediction = st.session_state['prediction']
        st.metric("Success Probability", f"{prediction*100:.1f}%")
        
        # Visual gauge
        fig3, ax3 = plt.subplots(figsize=(6, 2))
        ax3.barh(['Success'], [prediction*100], color='#4CAF50')
        ax3.set_xlim(0, 100)
        ax3.set_title("Performance Probability")
        st.pyplot(fig3)
        
        # Display input values
        st.subheader("📝 Input Summary")
        input_data = st.session_state['input_data']
        for key, value in input_data.items():
            st.write(f"{key}: {value}")
    else:
        st.info("Adjust the parameters in the sidebar and click 'Predict Performance'")

# Sample data table (mock)
st.subheader("📋 Sample Student Data")
sample_data = pd.DataFrame({
    'Student_ID': range(1, 6),
    'Attendance': np.random.randint(60, 100, 5),
    'Assignment_Score': np.random.randint(50, 100, 5),
    'Midterm_Score': np.random.randint(40, 95, 5),
    'Final_Score': np.random.randint(30, 100, 5),
    'Outstanding_Balance': np.random.randint(0, 2000, 5)
})
st.dataframe(sample_data)

