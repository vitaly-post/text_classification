FROM ubuntu:18.04

RUN apt-get update -y
RUN apt-get install python3.7 -y
RUN apt-get install python3-pip -y
RUN python3.7 --version
RUN apt-get install nano -y

WORKDIR /app
COPY . /app

RUN pip3 install --upgrade pip

RUN pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

RUN pip install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["./app.py"]