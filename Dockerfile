FROM tleyden5iwx/caffe-cpu-master

ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
CMD python app.py
