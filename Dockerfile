FROM python:3.10-slim  

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    ffmpeg

COPY . /app

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip install -r requirements.txt

CMD ["python", "toothfairy_train.py"]
