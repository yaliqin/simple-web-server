FROM tensorflow/tensorflow:2.4.1-gpu
EXPOSE 80
WORKDIR /project
ADD . /project
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
