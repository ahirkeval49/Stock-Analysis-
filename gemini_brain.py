from google import genai
import streamlit as st

def get_gemini_client():
    # Make sure GEMINI_API_KEY is in secrets.toml
    return genai.Client(api_key=st.secrets["API_KEY"])

def generate_agent_analysis(agent_role, data_context, prompt_instruction):
    client = get_gemini_client()
    
    full_prompt = f"""
    ROLE: {agent_role}
    
    DATA CONTEXT:
    {data_context}
    
    INSTRUCTION:
    {prompt_instruction}
    
    FORMAT:
    Provide a concise analysis in Markdown. Use bullet points. Do not include introductory filler text.
    """
    
    try:
        # Using Gemini 2.0 Flash for speed and efficiency [cite: 75]
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        return f"Agent Error: {str(e)}"
