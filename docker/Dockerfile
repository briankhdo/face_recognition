FROM tensorflow/tensorflow:latest

RUN apt-get update -y --fix-missing
RUN apt-get install -y ffmpeg
RUN apt-get install -y build-essential cmake pkg-config \
                    libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
                    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
                    libxvidcore-dev libx264-dev \
                    libgtk-3-dev \
                    libatlas-base-dev gfortran \
                    libboost-all-dev \
                    python3 python3-dev python3-numpy

RUN apt-get install -y wget vim python3-tk python3-pip

RUN apt-get install git

RUN cd /root/ && git clone https://github.com/briankhdo/face_recognition.git

CMD ["python3 /root/face_recognition/app.py"]
