import os
import sys
import streamlit.web.cli as stcli

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    # Launch the application
    sys.argv = ["streamlit", "run", "src/main.py", "--server.enableCORS=false"]
    sys.exit(stcli.main())