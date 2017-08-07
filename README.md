## Real Time Facial Emotion Analyzer ##

### Installing dependencies ###

* Install OpenCV 2.4 using your preferred method (e.g. `brew install opencv` on OSX)
* Install TensorFlow: `pip install tensorflow-gpu --ignore-installed` then give it a try: `python -c "import tensorflow;print(tensorflow.__version__)"`
* Install Keras: `pip install keras` then give it a try: `python -c "import keras;print(keras.__version__)"`

In case you get h5py related errors, upgrade the package: `pip install h5py --upgrade`
Generally, if you get other install errors or have existing tensorflow/keras installs with different versions, use a [virtual environment](docs.python-guide.org/en/latest/dev/virtualenvs/)

Tested on OSX 10.11.5 with tensorflow 1.1.0, keras 2.0.6, h5py 2.7.0, cuda 8.0, cudnn 5.1.10

### Running ###

Run the program in terminal:
```
python real-time.py haarcascade_frontalface_default.xml
```