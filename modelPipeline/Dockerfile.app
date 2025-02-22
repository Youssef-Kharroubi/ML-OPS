FROM python:3.11.5
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py model_pipline.py ./
EXPOSE 5000
CMD ["python3", "main.py", "--load_model", "models/RF.pkl", "--port", "5000"]