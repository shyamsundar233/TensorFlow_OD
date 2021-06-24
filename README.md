# Tensor Flow 2.0 Object Detection

## Tensorflow installation

Install tensorflow 2.0 in ubuntu:  https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/
Install tensorflow 2.0 in macOS: https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/
    
## Usage

** Using terminal, from your root directory ``` TensorFlowOD/```,

** Start training the model:  

                ```$ python train.py```

** It will save the detector.h5 , plot.png , test_images.txt

> ```detector.h5``` : Serialized model after training

> ```plot.png``` : Training history plot graph

> ```test_images.txt``` : Contains the names of the images in our testing set

** Then , test or make prediction using the previously trained model:

                ```$ python predict.py --input [either the path of a single image or test_imges.txt]```
