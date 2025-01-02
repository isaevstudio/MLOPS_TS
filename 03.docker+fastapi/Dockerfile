
FROM python:3.12.1-slim

WORKDIR /app

# As i am using the pipenv
COPY Pipfile Pipfile.lock ./

RUN pip install --no-cache-dir pipenv

RUN pipenv install --system --deploy --ignore-pipfile

COPY ./ /app

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]

