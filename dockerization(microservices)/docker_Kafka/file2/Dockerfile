FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY file2.py .
CMD ["python", "file2.py"]
