FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY file1.py .
CMD ["python", "file1.py"]
