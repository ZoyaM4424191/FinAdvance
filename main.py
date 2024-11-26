import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import pdfplumber
from fpdf import FPDF
import openai
# Replace with your actual API key

# App Title
st.title("Financial Document Analysis with Growth, Anomaly Detection, and AI Insights")

# Helper Function to Extract Table from PDF
def extract_pdf_table(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                df = pd.DataFrame(tables[0][1:], columns=tables[0][0])
                return df
    return None

# Helper Function to Clean and Normalize Column Names
def clean_column_names(df):
    df.columns = df.columns.str.strip()
    return df

# Helper Function to Calculate Growth Insights
def calculate_growth(df, amount_column):
    df[amount_column] = pd.to_numeric(df[amount_column], errors='coerce')
    df['Growth Rate'] = df[amount_column].pct_change() * 100
    return df

# Helper Function for Threshold-Based Anomaly Detection
def detect_anomalies(df, amount_column, lower_threshold, upper_threshold):
    df['Anomaly'] = (df[amount_column] < lower_threshold) | (df[amount_column] > upper_threshold)
    anomalies = df[df['Anomaly']]
    return anomalies

# Function to Generate AI Insights using OpenAI
def generate_ai_insights(data, anomalies):
    data_summary = data.describe().to_string()
    anomaly_summary = anomalies.to_string(index=False) if not anomalies.empty else "No anomalies detected."
    
    prompt = f"""
    You are a financial analyst. Analyze the following financial data:
    
    Summary Statistics:
    {data_summary}
    
    Detected Anomalies:
    {anomaly_summary}
    
    Provide insights on trends, anomalies, and any recommendations based on the data.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial analyst providing detailed insights."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# Helper Function to Generate PDF Report
def generate_pdf_report(report_data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, "Financial Analysis Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    for section, content in report_data.items():
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(0, 10, section, ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, content)

    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer, 'S').encode('latin1')
    pdf_buffer.seek(0)
    return pdf_buffer

# Streamlit File Upload Section
uploaded_file = st.file_uploader("Upload a financial document (CSV, Excel, or PDF)", type=["csv", "xlsx", "pdf"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".pdf"):
            df = extract_pdf_table(uploaded_file)
        
        if df is not None:
            df = clean_column_names(df)

            st.markdown("### Uploaded Data Preview:")
            st.dataframe(df)

            # Select Column for Analysis
            amount_column = st.selectbox("Select the column to analyze for growth and anomalies:", df.columns)

            if amount_column:
                # Calculate Growth Insights
                df = calculate_growth(df, amount_column)

                # User-Defined Thresholds for Anomaly Detection
                st.markdown("### Set Anomaly Detection Thresholds")
                lower_threshold = st.number_input("Enter lower threshold:", value=df[amount_column].min())
                upper_threshold = st.number_input("Enter upper threshold:", value=df[amount_column].max())

                # Detect Anomalies
                anomalies = detect_anomalies(df, amount_column, lower_threshold, upper_threshold)

                st.markdown("### Growth Insights:")
                st.dataframe(df[[amount_column, 'Growth Rate']].style.format({'Growth Rate': "{:.2f}%"}))

                # Plot Growth and Anomalies
                fig, ax = plt.subplots()
                ax.plot(df.index, df[amount_column], label="Values")
                ax.axhline(y=lower_threshold, color='red', linestyle='--', label="Lower Threshold")
                ax.axhline(y=upper_threshold, color='green', linestyle='--', label="Upper Threshold")
                ax.set_title('Growth and Threshold-Based Anomalies')
                ax.set_ylabel(amount_column)
                ax.legend()
                st.pyplot(fig)

                # Display Anomalies
                if not anomalies.empty:
                    st.markdown("### Anomalies Detected:")
                    st.dataframe(anomalies)
                    st.warning(f"Detected {len(anomalies)} anomalies based on the thresholds.")
                else:
                    st.success("No anomalies detected based on the thresholds.")

                # Generate AI Insights
                with st.spinner("Generating AI Insights..."):
                    ai_insights = generate_ai_insights(df, anomalies)
                st.markdown("### AI-Generated Insights:")
                st.write(ai_insights)

                # PDF Report
                report_data = {
                    "Overview": "Financial performance analysis with growth and threshold-based anomaly detection.",
                    "Growth Insights": "Growth rates and detected anomalies are included in the analysis.",
                    "Anomalies": anomalies.to_string(index=False) if not anomalies.empty else "No anomalies detected.",
                    "AI Insights": ai_insights
                }
                pdf_report = generate_pdf_report(report_data)

                st.download_button("Download PDF Report", data=pdf_report, file_name="financial_report_with_ai_insights.pdf", mime="application/pdf")
            else:
                st.error("Please select a column for analysis.")
        else:
            st.error("No data extracted from the document.")
    except Exception as e:
        st.error(f"Error: {e}")

