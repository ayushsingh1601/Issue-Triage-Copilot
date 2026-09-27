FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY triage ./triage
COPY webapp ./webapp

RUN pip install --no-cache-dir .[web,trace] && \
    rm -rf /root/.cache/pip

ENV DATA_DIR=/data
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 7860

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')" || exit 1

CMD ["streamlit", "run", "webapp/app.py"]