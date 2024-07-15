FROM python:3.8-slim-buster

RUN apt update -y && apt-get install -y gcc
WORKDIR /app

COPY . /app
RUN pip install --upgrade pip

RUN pip --default-timeout=100 install -r requirements.txt

CMD ["python3", "app.py"]
