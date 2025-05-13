import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64

# Set page config
st.set_page_config(
    page_title="Jaya Jaya Institut - Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)


# Load the trained model, preprocessor, and feature names
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('optimal_threshold.pkl', 'rb') as f:
        optimal_threshold = pickle.load(f)
    return model, preprocessor, feature_names, optimal_threshold


model, preprocessor, feature_names, optimal_threshold = load_model()


# Function to create a sample dataset for batch prediction
@st.cache_data
def load_sample_data():
    # Try to load the sample data, if it exists
    try:
        return pd.read_csv('data.csv', sep=';')
    except FileNotFoundError:
        st.error("Sample data file not found. Please upload a CSV file with student data.")
        return None


# Function for prediction
def predict_dropout_risk(input_data, threshold=0.5):
    # Make prediction
    proba = model.predict_proba(input_data)[0][1]
    prediction = 1 if proba >= threshold else 0

    # Return results
    return {
        'risk_probability': proba,
        'prediction': prediction,
        'status': 'At Risk' if prediction == 1 else 'Not At Risk',
        'risk_level': get_risk_level(proba)
    }


def get_risk_level(probability):
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"


def get_recommendation(risk_level, features):
    # Low academic performance
    low_academic = features['Curricular_units_1st_sem_grade'] < 12 or features['Curricular_units_2nd_sem_grade'] < 12

    # Financial issues
    financial_issues = features['Debtor'] == 1 or features['Tuition_fees_up_to_date'] == 0

    # Low admission grade
    low_admission = features['Admission_grade'] < 120

    # Age factor
    mature_student = features['Age_at_enrollment'] > 25

    # Generate recommendations
    recommendations = []

    if risk_level == "Low Risk":
        recommendations.append("Regular check-ins with academic advisor")
        if low_academic:
            recommendations.append("Consider optional tutoring sessions")

    elif risk_level == "Medium Risk":
        if low_academic:
            recommendations.append("Schedule mandatory tutoring sessions")
            recommendations.append("Develop a study plan with academic advisor")
        if financial_issues:
            recommendations.append("Explore scholarship opportunities")
            recommendations.append("Schedule meeting with financial aid office")
        if mature_student:
            recommendations.append("Connect with mature student support group")

    elif risk_level == "High Risk":
        recommendations.append("Immediate intervention required")
        if low_academic:
            recommendations.append("Intensive academic support program")
            recommendations.append("Consider reduced course load")
        if financial_issues:
            recommendations.append("Urgent financial counseling")
            recommendations.append("Payment plan assessment")
        if low_admission:
            recommendations.append("Foundational skills assessment")
            recommendations.append("Specialized tutoring for core subjects")
        if mature_student:
            recommendations.append("Work-study balance counseling")

    return recommendations


# Custom CSS
def add_custom_style():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .risk-high {
        font-weight: bold;
        color: #CF2E2E;
    }
    .risk-medium {
        font-weight: bold;
        color: #FF9900;
    }
    .risk-low {
        font-weight: bold;
        color: #28A745;
    }
    .dashboard-link {
        text-align: center;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    add_custom_style()

    # App header
    st.markdown("<h1 class='main-header'>🎓 Jaya Jaya Institut</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Student Dropout Prediction System</h2>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Individual Prediction", "Batch Prediction", "About", "Documentation"])

    with tab1:
        st.markdown("<h3 class='sub-header'>Individual Student Risk Assessment</h3>", unsafe_allow_html=True)
        st.write("Enter student information to predict dropout risk.")

        # Create columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            # Demographic information
            st.subheader("Demographic Information")
            age = st.number_input("Age at Enrollment", min_value=17, max_value=70, value=19)
            gender = st.selectbox("Gender", options=["Male", "Female"])
            marital_status = st.selectbox("Marital Status",
                                          options=["Single", "Married", "Widower", "Divorced", "Separated",
                                                   "Facto union"])
            nationality = st.selectbox("Nationality", options=["Portuguese", "Foreign"])
            displaced = st.selectbox("Displaced from Home", options=["Yes", "No"])

            # Special needs and financial factors
            st.subheader("Special Needs & Financial Status")
            special_needs = st.selectbox("Educational Special Needs", options=["Yes", "No"])
            scholarship = st.selectbox("Scholarship Holder", options=["Yes", "No"])
            debtor = st.selectbox("Debtor", options=["Yes", "No"])
            tuition_up_to_date = st.selectbox("Tuition Fees Up to Date", options=["Yes", "No"])

        with col2:
            # Academic background
            st.subheader("Academic Background")
            application_mode = st.selectbox("Application Mode",
                                            options=["1st phase - general contingent", "Ordinance No. 612/93",
                                                     "1st phase - special contingent (Azores Island)",
                                                     "Holders of other higher courses", "Ordinance No. 854-B/99",
                                                     "2nd phase - general contingent", "3rd phase - general contingent",
                                                     "Over 23 years old", "Transfer", "Change of course",
                                                     "Technological specialization diploma holders",
                                                     "Change of institution/course", "International student (bachelor)",
                                                     "1st phase - special contingent (Madeira Island)",
                                                     "Ordinance No. 533-A/99, item b2) (Graduates)",
                                                     "Ordinance No. 533-A/99, item b3 (Graduates)"], index=0)
            previous_qualification = st.selectbox("Previous Qualification", options=["Secondary education",
                                                                                     "Higher education - bachelor's degree",
                                                                                     "Higher education - degree",
                                                                                     "Higher education - master's degree",
                                                                                     "Higher education - doctorate",
                                                                                     "Frequency of higher education",
                                                                                     "12th year of schooling - not completed",
                                                                                     "11th year of schooling - not completed",
                                                                                     "Other - 11th year of schooling",
                                                                                     "10th year of schooling",
                                                                                     "10th year of schooling - not completed",
                                                                                     "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
                                                                                     "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
                                                                                     "Technological specialization course",
                                                                                     "Higher education - degree (1st cycle)",
                                                                                     "Professional higher technical course",
                                                                                     "Higher education - master's degree (2nd cycle)"],
                                                  index=0)
            qualification_grade = st.number_input("Previous Qualification Grade (0-200)", min_value=0, max_value=200,
                                                  value=130)
            admission_grade = st.number_input("Admission Grade (0-200)", min_value=0, max_value=200, value=135)

            # Course information
            st.subheader("Course Performance")
            course_code = st.number_input("Course Code", min_value=1, max_value=17, value=9)
            daytime_evening = st.selectbox("Daytime/Evening Attendance", options=["Daytime", "Evening"])

            # First semester
            units_1st_enrolled = st.number_input("Curricular Units 1st Sem Enrolled", min_value=0, max_value=12,
                                                 value=6)
            units_1st_approved = st.number_input("Curricular Units 1st Sem Approved", min_value=0, max_value=12,
                                                 value=5)
            units_1st_grade = st.number_input("Curricular Units 1st Sem Grade (0-20)", min_value=0, max_value=20,
                                              value=13)
            units_1st_evaluations = st.number_input("Curricular Units 1st Sem Evaluations", min_value=0, max_value=30,
                                                    value=6)

            # Second semester
            units_2nd_enrolled = st.number_input("Curricular Units 2nd Sem Enrolled", min_value=0, max_value=12,
                                                 value=6)
            units_2nd_approved = st.number_input("Curricular Units 2nd Sem Approved", min_value=0, max_value=12,
                                                 value=5)
            units_2nd_grade = st.number_input("Curricular Units 2nd Sem Grade (0-20)", min_value=0, max_value=20,
                                              value=13)
            units_2nd_evaluations = st.number_input("Curricular Units 2nd Sem Evaluations", min_value=0, max_value=30,
                                                    value=6)

        # Economic factors
        st.subheader("Economic Factors")
        col_eco1, col_eco2, col_eco3 = st.columns(3)
        with col_eco1:
            unemployment_rate = st.number_input("Unemployment Rate", min_value=0.0, max_value=100.0, value=10.8)
        with col_eco2:
            inflation_rate = st.number_input("Inflation Rate", min_value=-5.0, max_value=20.0, value=1.4)
        with col_eco3:
            gdp = st.number_input("GDP", min_value=-10.0, max_value=10.0, value=0.8)

        # Convert categorical variables to the format expected by the model
        gender_binary = 1 if gender == "Male" else 0
        displaced_binary = 1 if displaced == "Yes" else 0
        special_needs_binary = 1 if special_needs == "Yes" else 0
        scholarship_binary = 1 if scholarship == "Yes" else 0
        debtor_binary = 1 if debtor == "Yes" else 0
        tuition_up_to_date_binary = 1 if tuition_up_to_date == "Yes" else 0
        international_binary = 1 if nationality == "Foreign" else 0
        daytime_binary = 1 if daytime_evening == "Daytime" else 0

        # Create features dictionary
        features = {
            'Marital_status': marital_status,
            'Application_mode': application_mode,
            'Application_order': 1,  # Default value
            'Course': course_code,
            'Daytime_evening_attendance': daytime_binary,
            'Previous_qualification': previous_qualification,
            'Previous_qualification_grade': qualification_grade,
            'Nacionality': nationality,
            'Mothers_qualification': "Secondary Education",  # Default value
            'Fathers_qualification': "Secondary Education",  # Default value
            'Mothers_occupation': 7,  # Default value
            'Fathers_occupation': 7,  # Default value
            'Admission_grade': admission_grade,
            'Displaced': displaced_binary,
            'Educational_special_needs': special_needs_binary,
            'Debtor': debtor_binary,
            'Tuition_fees_up_to_date': tuition_up_to_date_binary,
            'Gender': gender_binary,
            'Scholarship_holder': scholarship_binary,
            'Age_at_enrollment': age,
            'International': international_binary,
            'Curricular_units_1st_sem_credited': 0,  # Default value
            'Curricular_units_1st_sem_enrolled': units_1st_enrolled,
            'Curricular_units_1st_sem_evaluations': units_1st_evaluations,
            'Curricular_units_1st_sem_approved': units_1st_approved,
            'Curricular_units_1st_sem_grade': units_1st_grade,
            'Curricular_units_1st_sem_without_evaluations': units_1st_enrolled - units_1st_evaluations,
            'Curricular_units_2nd_sem_credited': 0,  # Default value
            'Curricular_units_2nd_sem_enrolled': units_2nd_enrolled,
            'Curricular_units_2nd_sem_evaluations': units_2nd_evaluations,
            'Curricular_units_2nd_sem_approved': units_2nd_approved,
            'Curricular_units_2nd_sem_grade': units_2nd_grade,
            'Curricular_units_2nd_sem_without_evaluations': units_2nd_enrolled - units_2nd_evaluations,
            'Unemployment_rate': unemployment_rate,
            'Inflation_rate': inflation_rate,
            'GDP': gdp
        }

        # Create a DataFrame from the features
        input_df = pd.DataFrame([features])

        # Make prediction when button is clicked
        if st.button("Predict Dropout Risk"):
            # Calculate additional metrics for display
            academic_success_rate = (units_1st_approved + units_2nd_approved) / (
                        units_1st_enrolled + units_2nd_enrolled) if (units_1st_enrolled + units_2nd_enrolled) > 0 else 0
            avg_grade = (units_1st_grade + units_2nd_grade) / 2

            # Make prediction
            result = predict_dropout_risk(input_df, threshold=optimal_threshold)

            # Display result
            st.markdown("---")
            st.subheader("Prediction Result")

            # Create columns for better layout
            col_result1, col_result2 = st.columns(2)

            with col_result1:
                # Risk assessment card
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; border: 1px solid #ddd; background-color: #1c1c1e;">
                    <h3 style="text-align: center; margin-bottom: 15px;">Risk Assessment</h3>
                    <p style="font-size: 18px; text-align: center;">
                        Risk Probability: <strong>{result['risk_probability']:.2%}</strong>
                    </p>
                    <p style="font-size: 24px; text-align: center;" class="risk-{'low' if result['risk_level'] == 'Low Risk' else 'medium' if result['risk_level'] == 'Medium Risk' else 'high'}">
                        {result['risk_level']}
                    </p>
                    <p style="font-size: 18px; text-align: center;">
                        Status: <strong>{result['status']}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with col_result2:
                # Student metrics card
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; border: 1px solid #ddd; background-color: #1c1c1e;">
                    <h3 style="text-align: center; margin-bottom: 15px;">Student Metrics</h3>
                    <p style="font-size: 16px;">
                        <strong>Academic Success Rate:</strong> {academic_success_rate:.2%}
                    </p>
                    <p style="font-size: 16px;">
                        <strong>Average Grade:</strong> {avg_grade:.1f}/20
                    </p>
                    <p style="font-size: 16px;">
                        <strong>Financial Status:</strong> {"At Risk" if debtor_binary == 1 or tuition_up_to_date_binary == 0 else "Good Standing"}
                    </p>
                    <p style="font-size: 16px;">
                        <strong>Scholarship Status:</strong> {"Has Scholarship" if scholarship_binary == 1 else "No Scholarship"}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Recommendations section
            st.markdown("### Recommended Actions")
            recommendations = get_recommendation(result['risk_level'], features)

            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}. {rec}**")

            # Visualization: Gauge for risk probability
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Dropout Risk Probability')
            ax.add_patch(plt.Rectangle((0, 0), 0.3, 1, alpha=0.3, color='green'))
            ax.add_patch(plt.Rectangle((0.3, 0), 0.3, 1, alpha=0.3, color='orange'))
            ax.add_patch(plt.Rectangle((0.6, 0), 0.4, 1, alpha=0.3, color='red'))
            ax.axvline(x=result['risk_probability'], color='blue', linestyle='-', linewidth=4)
            ax.text(0.15, 0.5, 'Low Risk', ha='center', va='center', fontsize=12)
            ax.text(0.45, 0.5, 'Medium Risk', ha='center', va='center', fontsize=12)
            ax.text(0.8, 0.5, 'High Risk', ha='center', va='center', fontsize=12)
            ax.text(result['risk_probability'], 0.85, f"{result['risk_probability']:.2%}",
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            ax.set_axis_off()
            st.pyplot(fig)

    with tab2:
        st.markdown("<h3 class='sub-header'>Batch Prediction</h3>", unsafe_allow_html=True)
        st.write("Upload a CSV file with student data to predict dropout risk for multiple students.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file, sep=';')
                st.success(f"Successfully loaded file with {len(batch_data)} students.")

                if st.button("Run Batch Prediction"):
                    # Check if Status column exists and remove it for prediction
                    if 'Status' in batch_data.columns:
                        X_batch = batch_data.drop('Status', axis=1)
                    else:
                        X_batch = batch_data.copy()

                    # Make predictions
                    with st.spinner("Processing predictions..."):
                        batch_proba = model.predict_proba(X_batch)[:, 1]
                        batch_predictions = (batch_proba >= optimal_threshold).astype(int)

                        # Add predictions to the dataframe
                        results_df = batch_data.copy()
                        results_df['Dropout_Risk_Probability'] = batch_proba
                        results_df['Risk_Level'] = [get_risk_level(p) for p in batch_proba]
                        results_df['Predicted_Status'] = ['At Risk' if p == 1 else 'Not At Risk' for p in
                                                          batch_predictions]

                        # Display results
                        st.subheader("Prediction Results")

                        # Display metrics
                        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)

                        with col_metrics1:
                            high_risk_count = sum(results_df['Risk_Level'] == 'High Risk')
                            high_risk_percentage = high_risk_count / len(results_df) * 100
                            st.metric("High Risk Students", high_risk_count, f"{high_risk_percentage:.1f}%")

                        with col_metrics2:
                            medium_risk_count = sum(results_df['Risk_Level'] == 'Medium Risk')
                            medium_risk_percentage = medium_risk_count / len(results_df) * 100
                            st.metric("Medium Risk Students", medium_risk_count, f"{medium_risk_percentage:.1f}%")

                        with col_metrics3:
                            low_risk_count = sum(results_df['Risk_Level'] == 'Low Risk')
                            low_risk_percentage = low_risk_count / len(results_df) * 100
                            st.metric("Low Risk Students", low_risk_count, f"{low_risk_percentage:.1f}%")

                        # Distribution plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(results_df['Dropout_Risk_Probability'], bins=20, kde=True, ax=ax)
                        ax.axvline(x=optimal_threshold, color='red', linestyle='--',
                                   label=f'Threshold ({optimal_threshold:.2f})')
                        ax.set_title('Distribution of Dropout Risk Probabilities')
                        ax.set_xlabel('Risk Probability')
                        ax.set_ylabel('Count')
                        ax.legend()
                        st.pyplot(fig)

                        # Display dataframe with ability to sort by risk
                        st.subheader("Student Risk Table")
                        st.dataframe(results_df.sort_values('Dropout_Risk_Probability', ascending=False))

                        # Option to download results
                        csv = results_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="dropout_risk_predictions.csv">Download Prediction Results (CSV)</a>'
                        st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Please make sure the CSV file has the same structure as the training data.")

        # If no file uploaded, offer sample data option
        else:
            st.info("No file uploaded. You can use sample data for demonstration.")
            if st.button("Use Sample Data"):
                sample_data = load_sample_data()
                if sample_data is not None:
                    st.success(f"Loaded sample data with {len(sample_data)} students.")
                    st.write("Preview of sample data:")
                    st.dataframe(sample_data.head())

                    if st.button("Run Prediction on Sample"):
                        # Similar batch prediction as above
                        with st.spinner("Processing predictions..."):
                            # Remove Status column if exists
                            if 'Status' in sample_data.columns:
                                X_sample = sample_data.drop('Status', axis=1)
                            else:
                                X_sample = sample_data.copy()

                            sample_proba = model.predict_proba(X_sample)[:, 1]
                            sample_predictions = (sample_proba >= optimal_threshold).astype(int)

                            # Add predictions to the dataframe
                            results_df = sample_data.copy()
                            results_df['Dropout_Risk_Probability'] = sample_proba
                            results_df['Risk_Level'] = [get_risk_level(p) for p in sample_proba]
                            results_df['Predicted_Status'] = ['At Risk' if p == 1 else 'Not At Risk' for p in
                                                              sample_predictions]

                            # Display metrics (same as above)
                            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)

                            with col_metrics1:
                                high_risk_count = sum(results_df['Risk_Level'] == 'High Risk')
                                high_risk_percentage = high_risk_count / len(results_df) * 100
                                st.metric("High Risk Students", high_risk_count, f"{high_risk_percentage:.1f}%")

                            with col_metrics2:
                                medium_risk_count = sum(results_df['Risk_Level'] == 'Medium Risk')
                                medium_risk_percentage = medium_risk_count / len(results_df) * 100
                                st.metric("Medium Risk Students", medium_risk_count, f"{medium_risk_percentage:.1f}%")

                            with col_metrics3:
                                low_risk_count = sum(results_df['Risk_Level'] == 'Low Risk')
                                low_risk_percentage = low_risk_count / len(results_df) * 100
                                st.metric("Low Risk Students", low_risk_count, f"{low_risk_percentage:.1f}%")

                            # Distribution plot (same as above)
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(results_df['Dropout_Risk_Probability'], bins=20, kde=True, ax=ax)
                            ax.axvline(x=optimal_threshold, color='red', linestyle='--',
                                       label=f'Threshold ({optimal_threshold:.2f})')
                            ax.set_title('Distribution of Dropout Risk Probabilities')
                            ax.set_xlabel('Risk Probability')
                            ax.set_ylabel('Count')
                            ax.legend()
                            st.pyplot(fig)

                            # Display dataframe with ability to sort by risk
                            st.subheader("Student Risk Table")
                            st.dataframe(results_df.sort_values('Dropout_Risk_Probability', ascending=False))

    with tab3:
        st.markdown("<h3 class='sub-header'>About This Project</h3>", unsafe_allow_html=True)

        st.markdown("""
        ### Project Overview

        This application was developed to help Jaya Jaya Institut identify students at risk of dropping out. 
        By predicting which students might drop out early, the institution can provide targeted support to 
        improve retention rates and student success.

        ### Key Features

        - **Individual Student Assessment**: Predict dropout risk for a single student
        - **Batch Prediction**: Analyze dropout risk for multiple students at once
        - **Risk Categorization**: Classify students as Low, Medium, or High risk
        - **Personalized Recommendations**: Get tailored intervention strategies based on student profiles
        - **Interactive Dashboard**: Visual analytics for monitoring student performance

        ### Methodology

        The prediction model was trained on historical student data, analyzing various factors that 
        influence dropout rates. The system uses machine learning to identify patterns and predict 
        future outcomes with high accuracy.

        ### Dashboard Integration

        This application is integrated with a Metabase dashboard for more comprehensive analytics. 
        The dashboard provides visualizations of student performance metrics and dropout risk factors.

        [Access Dashboard](#) (Access requires authentication)
        """)

        st.markdown(
            "<div class='dashboard-link'><a href='http://localhost:3000' target='_blank'>Access Metabase Dashboard</a></div>",
            unsafe_allow_html=True)

    with tab4:
        st.markdown("<h3 class='sub-header'>Documentation</h3>", unsafe_allow_html=True)

        st.markdown("""
        ### User Guide

        #### Individual Prediction
        1. Navigate to the "Individual Prediction" tab
        2. Fill in the student information form
        3. Click "Predict Dropout Risk"
        4. Review the risk assessment and recommendations

        #### Batch Prediction
        1. Navigate to the "Batch Prediction" tab
        2. Upload a CSV file with student data
        3. Click "Run Batch Prediction"
        4. Review the results and download if needed

        ### Data Requirements

        The batch prediction CSV file should include the following columns:
        - Demographic information (Age, Gender, etc.)
        - Academic performance metrics
        - Financial status indicators
        - Course enrollment details

        For a complete list of required columns, download the [sample template](#).

        ### Model Information

        - **Algorithm**: Gradient Boosting Classifier
        - **Accuracy**: ~85% on test data
        - **Key Predictors**: First semester performance, financial status, admission grade

        ### Integration with Metabase

        The system is integrated with Metabase for advanced analytics. To access the dashboard:
        1. Use the link in the "About" tab
        2. Login with provided credentials
        3. Navigate to the "Student Performance" dashboard
        """)

    # Footer
    st.markdown("<div class='footer'>© 2025 Jaya Jaya Institut - Student Success Project</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()