FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY agentic_os ./agentic_os
RUN pip install --no-cache-dir ".[anthropic,gemini]" && useradd --create-home agent
USER agent

ENV AGENTIC_OS_DATA_DIR=/home/agent/data
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["agentic-os", "serve", "--host", "0.0.0.0", "--port", "8000"]
