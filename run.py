import os
import sys
import streamlit.web.cli as stcli

# Добавляем корень проекта в путь Python
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    # Запускаем приложение
    sys.argv = ["streamlit", "run", "src/main.py", "--server.enableCORS=false"]
    sys.exit(stcli.main())