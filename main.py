import json
import tempfile
import csv
import streamlit as st
import pandas as pd
import re
import duckdb
import matplotlib.pyplot as plt

from agno.models.openai import OpenAIChat
from phi.agent.duckdb import DuckDbAgent
from agno.tools.pandas import PandasTools

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# Streamlit UI
st.set_page_config(page_title="📊 InsightGenie: AI-Powered Data Analyst", layout="wide")
st.title("📊 InsightGenie: AI-Powered Data Analyst")

# Sidebar API input
with st.sidebar:
    st.header("🔐 API Configuration")
    openai_key = st.text_input("Enter your OpenAI API key:", type="password")
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("✅ API key saved!")
    else:
        st.warning("Please enter your OpenAI API key to continue.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file and "openai_key" in st.session_state:
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path and columns and df is not None:
        st.subheader("📄 Preview of Uploaded Data")
        st.dataframe(df)

        st.subheader("📚 Available Columns")
        st.write(columns)

        semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                    "path": temp_path,
                }
            ]
        }

        duckdb_agent = DuckDbAgent(
            model=OpenAIChat(model="gpt-4", api_key=st.session_state.openai_key),
            semantic_model=json.dumps(semantic_model),
            tools=[PandasTools()],
            markdown=True,
            add_history_to_messages=False,
            followups=False,
            read_tool_call_history=False,
            system_prompt="You are an expert data analyst. Generate SQL queries to solve the user's query. Return only the SQL query, enclosed in ```sql ``` and give the final answer.",
        )

        if "generated_code" not in st.session_state:
            st.session_state.generated_code = None

        st.subheader("🧠 Ask a Question About Your Data")
        user_query = st.text_area("E.g., What is the total sales by region?", height=100)
        st.info("💡 Tip: The SQL result and chart will appear below once you submit.")

        if st.button("Submit Query"):
            if not user_query.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    with st.spinner('🔍 Processing your query...'):
                        response1 = duckdb_agent.run(user_query)
                        response_content = response1.content if hasattr(response1, 'content') else str(response1)
                        response = duckdb_agent.print_response(user_query, stream=True)

                        # Extract SQL code block
                        sql_match = re.search(r"```sql\s+(.*?)```", response_content, re.DOTALL)
                        if sql_match:
                            sql_query = sql_match.group(1)

                            # Run the SQL using DuckDB
                            result_df = duckdb.sql(f"SELECT * FROM read_csv_auto('{temp_path}')").df()
                            query_result = duckdb.query(sql_query).to_df()

                            st.markdown("### 🧠 AI Response")
                            st.markdown(response_content)

                            st.markdown("### 📋 SQL Query Result")
                            st.dataframe(query_result)

                            # 📊 Visualization
                            st.markdown("### 📊 Visualize Your Result")
                            if query_result.shape[1] >= 2:
                                col1, col2 = st.columns(2)
                                with col1:
                                    x_axis = st.selectbox("X-axis", query_result.columns, key="x")
                                with col2:
                                    y_axis = st.selectbox("Y-axis", query_result.columns, key="y")

                                chart_type = st.radio("Chart Type", ["Bar", "Line", "Area", "Pie"], horizontal=True)

                                if chart_type == "Bar":
                                    st.bar_chart(query_result.set_index(x_axis)[y_axis])
                                elif chart_type == "Line":
                                    st.line_chart(query_result.set_index(x_axis)[y_axis])
                                elif chart_type == "Area":
                                    st.area_chart(query_result.set_index(x_axis)[y_axis])
                                elif chart_type == "Pie":
                                    fig, ax = plt.subplots()
                                    query_result.groupby(x_axis)[y_axis].sum().plot.pie(
                                        ax=ax, autopct='%1.1f%%', ylabel=''
                                    )
                                    st.pyplot(fig)
                            else:
                                st.info("Not enough columns to generate a chart. Try a broader query.")
                        else:
                            st.warning("⚠️ No SQL found in the AI response.")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.error("Please try rephrasing your query or check if the data format is valid.")
