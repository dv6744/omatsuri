FROM python:3.12-slim

WORKDIR /app

RUN pip install streamlit requests

COPY frontend/app.py ./frontend/app.py

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
