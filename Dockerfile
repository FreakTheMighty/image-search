FROM tleyden5iwx/caffe-cpu-master

ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
RUN git clone https://github.com/kayzh/LSHash.git; cd LSHash; python setup.py install
CMD python app.py
