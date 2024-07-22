FROM python:3.12.4-slim-bookworm as base

FROM base as builder

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /install/
COPY README.md /install/

WORKDIR /install

RUN pip install -r requirements.txt

FROM base

COPY --from=builder /opt/venv /opt/venv

COPY main.py /app/
COPY Main_model2.pkl /app/
COPY good_or_bad.pkl /app/

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
