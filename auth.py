import streamlit as st
from datetime import datetime

class KeyManager:
    def __init__(self, keys):
        self.keys = keys
        # Initialize usage state in session if not present
        if 'api_usage' not in st.session_state:
            st.session_state.api_usage = {key: 0 for key in keys}
            st.session_state.last_reset = datetime.now().date()
            
    def _check_reset(self):
        # Reset if new day (UTC check simplified here)
        if datetime.now().date() > st.session_state.last_reset:
            st.session_state.api_usage = {key: 0 for key in self.keys}
            st.session_state.last_reset = datetime.now().date()

    def get_active_key(self):
        self._check_reset()
        
        # Check all keys to see if any are under the limit (25 calls)
        for key in self.keys:
            if st.session_state.api_usage[key] < 25: 
                return key
        
        return None # All keys exhausted

    def log_request(self, key):
        if key in st.session_state.api_usage:
            st.session_state.api_usage[key] += 1
