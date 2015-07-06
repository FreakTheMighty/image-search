FROM tleyden5iwx/caffe-cpu-master

ADD . /code
WORKDIR /code
RUN git clone https://github.com/FreakTheMighty/LSHash.git; cd LSHash; python setup.py install
RUN pip install -r requirements.txt
CMD python app.py
