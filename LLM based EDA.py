import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from anthropic import Anthropic
import io
import contextlib
import traceback
import re
import base64

# --- Page Configuration ---
st.set_page_config(page_title="AI Data Analyst", layout="wide", page_icon="📊")

# --- Constants & Configuration ---
SYSTEM_PROMPT = """You are an expert Data Analyst and Data Scientist. 
You are tasked with analyzing a dataset using Python.
You will be provided with the dataset schema (columns, types) and a sample of rows, along with a user's natural language question.

Your job:
1. Explain your thought process briefly.
2. Provide ONE python code block to perform the analysis.

Available libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns), plotly.express (px), plotly.graph_objects (go).
The dataset is ALREADY loaded in the environment as a pandas DataFrame named `df`. DO NOT try to load it from a file.

Important Code Execution Rules:
- If you generate text output, table snippets, or numbers, use `print()`.
- NEVER use plt.show() or fig.show(). The code is running headlessly.
- If you create a visual chart, you MUST use Plotly (`px` or `go`).
- You MUST assign the final Plotly figure to a variable named EXACTLY `fig` so the UI can render it natively.
- Only output the necessary code inside a single ```python code block.
"""

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# --- Helper Functions ---
def execute_code(code, df):
    # Namespace for execution
    local_namespace = {
        'df': df.copy(), # Pass a copy to prevent destructive edits unless intended
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'px': px,
        'go': go
    }
    
    output_capture = io.StringIO()
    error_output = ""
    
    try:
        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
            exec(code, local_namespace)
    except Exception as e:
        error_output = traceback.format_exc()
        
    fig = local_namespace.get('fig', None)
    return output_capture.getvalue(), error_output, fig

def extract_code(text):
    pattern = r"```(?:python)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def auto_eda(df):
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Types & Missing Values")
        info_df = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum(),
            "% Missing": (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True)
        
    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(include='all').T if len(df.columns) > 0 else df, use_container_width=True)
        
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 1:
        st.subheader("Correlation Matrix")
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Feature Correlations")
        st.plotly_chart(fig, use_container_width=True)
    elif len(num_cols) == 1:
        st.subheader("Distribution")
        fig = px.histogram(df, x=num_cols[0], title=f"Distribution of {num_cols[0]}")
        st.plotly_chart(fig, use_container_width=True)

def get_download_link(text_content, file_name, link_text):
    b64 = base64.b64encode(text_content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{file_name}">{link_text}</a>'

# --- UI Layout ---

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key_input = st.text_input("Anthropic API Key", type="password", value=st.session_state.api_key)
    if api_key_input:
        st.session_state.api_key = api_key_input
        
    st.markdown("---")
    st.title("💬 Chat History")
    
    if st.session_state.messages:
        for idx, msg in enumerate(st.session_state.messages):
            if msg['role'] == 'user':
                with st.expander(f"Q: {msg['content'][:30]}...", expanded=False):
                    st.write(msg['content'])
                    if idx + 1 < len(st.session_state.messages):
                        response = st.session_state.messages[idx+1]
                        st.markdown("**A:** " + response['content'][:100] + "...")
                        
        st.markdown("---")
        if st.button("Export Conversation"):
            convo_text = ""
            for m in st.session_state.messages:
                convo_text += f"{m['role'].upper()}:\n{m['content']}\n\n"
                if 'code_output' in m:
                    convo_text += f"OUTPUT:\n{m['code_output']}\n\n"
            st.markdown(get_download_link(convo_text, "chat_history.txt", "📥 Download Chat as TXT"), unsafe_allow_html=True)
            
        if st.button("Clear History"):
            st.session_state.messages = []
            st.rerun()

# Main Application
st.title("🧠 AI Data Analyst Assistant")

# File Upload Module
if st.session_state.df is None:
    st.info("Please upload a dataset to begin the analysis.")
    uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            st.session_state.df = df
            st.success("File uploaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")
else:
    df = st.session_state.df
    st.success(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns.")
    
    if st.button("Upload a different file"):
        st.session_state.df = None
        st.session_state.messages = []
        st.rerun()

    # Tabs for Data and Chat
    tab1, tab2 = st.tabs(["📊 Automatic EDA", "💬 Chat Analyst"])
    
    with tab1:
        auto_eda(df)
        
    with tab2:
        # Render the chat display (except the last unrendered one if any)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Render code and outputs if this was an assistant message
                if msg.get("code"):
                    with st.expander("Show Generated Code", expanded=False):
                        st.code(msg["code"], language="python")
                        
                if msg.get("code_output"):
                    st.text(msg["code_output"])
                    
                if msg.get("error"):
                    st.error(msg["error"])
                    
        # Chat Input logic
        if question := st.chat_input("Ask a question about your data..."):
            if not st.session_state.api_key:
                st.error("Please enter your Anthropic API Key in the sidebar.")
                st.stop()
                
            # Render user message
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Build dataset context
            df_info = f"Columns: {list(df.columns)}\nTypes:\n{df.dtypes}\nHead:\n{df.head().to_csv(index=False)}"
            prompt = f"Dataset Context:\n{df_info}\n\nUser Question: {question}"
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        client = Anthropic(api_key=st.session_state.api_key)
                        
                        api_messages = []
                        # Provide recent chat history
                        for m in st.session_state.messages[-7:-1]:
                            api_messages.append({"role": m["role"], "content": m["content"]})
                            
                        api_messages.append({"role": "user", "content": prompt})

                        response = client.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=4096,
                            system=SYSTEM_PROMPT,
                            messages=api_messages,
                            temperature=0.2
                        )
                        
                        response_text = response.content[0].text
                        st.markdown(response_text)
                        
                        msg_data = {"role": "assistant", "content": response_text}
                        
                        code_to_exec = extract_code(response_text)
                        if code_to_exec:
                            st.write("---")
                            st.write("### ⚙️ Execution Sandbox")
                            with st.expander("Generated Code", expanded=True):
                                st.code(code_to_exec, language="python")
                                
                            out, err, fig = execute_code(code_to_exec, df)
                            
                            msg_data["code"] = code_to_exec
                            
                            if out:
                                st.write("**Output:**")
                                st.text(out)
                                msg_data["code_output"] = out
                                
                            if err:
                                st.error(err)
                                msg_data["error"] = err
                                
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                # Streamlit handles Plotly charts natively
                                
                        st.session_state.messages.append(msg_data)
                        
                    except Exception as e:
                        st.error(f"API Error: {str(e)}")
