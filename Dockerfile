FROM python:3.10-slim  

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip install -r requirements.txt

CMD ["python", "toothfairy_train.py"]
