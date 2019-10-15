# Ultra-Light-Fast-Generic-Face-Detector-1MB 

![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/27.jpg)
The model design is a real-time ultra-lightweight universal face detection model designed for **edge computing devices** or **low computing devices** (eg with ARM reasoning), designed to be used in low-power devices The face detection reasoning of real-time common scenes using ARM is also applicable to the mobile environment (Android & IOS) and PC environment (CPU & GPU). Has the following characteristics:

- In terms of model size, the default FP32 precision (.pth) file size is **1.1MB**, and the inference frame int8 is quantized to a size of **300KB**.
  - In terms of model calculation, the input resolution of 320x240 is only **90~109 MFlops**, which is light enough.
  - There are two versions of the model design, version-slim (slightly faster simplification), version-RFB (with the modified RFB module, higher precision).
  - Provides a pre-training model using the widerface training at 320x240 and 640x480 different input resolutions to better work in different application scenarios.
  - No special operator, support onnx export, easy to transplant reasoning.

## Tested the normal operating environment
- Ubuntu16.04、Ubuntu18.04
- Python3.6
- Pytorch1.2
- CUDA10.0 + CUDNN7.6

## Accuracy, speed, scene test, model size comparison

Training set used: [Retinaface](https://github.com/deepinsight/insightface/blob/master/RetinaFace/README.md )The cleaned widerface tag is provided with the widerface dataset to generate the VOC training set (PS: the following test results are all tested by myself, and the results may be partially different).


### Widerface test
 - 在WIDER FACE test Set test accuracy (single scale input resolution: **320*240**)
模型|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
libfacedetection v2|0.4 |0.04       |0.02
官方 Retinaface-Mobilenet-0.25 (Mxnet)   |0.745|0.553|0.232
version-slim|0.765     |0.662       |0.385
version-RFB|**0.784**     |**0.688**       |**0.418**


- 在WIDER FACE test Set test accuracy (single scale input resolution: **VGA 640*480**)

模型|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
libfacedetection v1|0.197    |0.199       |0.112
libfacedetection v2|0.2 |0.218       |0.147
官方 Retinaface-Mobilenet-0.25 (Mxnet)   |**0.879**|0.807|0.481
version-slim|0.769     |0.733       |0.486
version-RFB|0.851     |**0.81**       |**0.541**

### Terminal device inference speed


-Raspberry Pi 4B MNN Reasoning Test Time ** (Unit: ms)** (ARM/A72x4/1.5GHz/Input Resolution: **320x240** /int8 Quantization)

Model|1 core|2 core|3 core|4 core
------|--------|----------|--------|--------
libfacedetection v1|**28**    |**16**|**12**|9.7
官方 Retinaface-Mobilenet-0.25 (Mxnet)   |46|25|18.5|15
version-slim|29     |**16**       |**12**|**9.5**
version-RFB|35     |19.6       |14.8| 11

### Scene test
- A number of different scene videos are roughly valid for face detection quantity test (unit: one) (resolution: **VGA 640*480**, threshold 0.6):

Model | Subway station outdoor. MP4 (1 minute 43 seconds) | Subway station. MP4 (1 minute 13 seconds) | Subway station 2.MP4 (36 seconds) | Wanda Plaza outdoor. MP4 (1 minute 36 seconds) | Office. MP4 (1 minute 07 seconds)
------|--------|----------|--------|--------|--------
libfacedetection v1| 6599   |4571|1899|7490|2818
官方 Retinaface-Mobilenet-0.25 (Mxnet) |4415|4897|2026|7882|2557
version-RFB|**10339** |**10444** |**4017**|**13363**|**3403**

###Model size comparison
- Comparison of the size of several mainstream open source lightweight face detection models:

Model|Model File Size (MB)
------|--------
libfacedetection v1（caffe）| 2.58
libfacedetection v2（caffe）| 3.34
官方 Retinaface-Mobilenet-0.25 (Mxnet) | 1.68
version-slim| **1.04**
version-RFB| **1.11** 

## Generate VOC format training data sets and training processes

1. Download the wideface official website dataset or download the training set I provided and extract it into the ./data folder:


  (1) Filter out the clean widerface data compression package after 10px*10px face: [Baiyun cloud disk extraction code: x5gt] (https://pan.baidu.com/s/1m600pp-AsNot6XgIiqDlOw)

  
  (2) The complete wideface data compression package of unfiltered small faces: [Baiyun cloud disk extraction code: 8748] (https://pan.baidu.com/s/1ijvZFSb3l7C63Nbz7i6IuQ)
 
 
2. ** (PS: If you download the filtered packets in (1) above, you don't need to perform this step)** Because there are many small and unclear faces in the widerface, it is not conducive to efficient models. Convergence, so you need to filter, the default is to filter the face size of 10 pixels * 10 pixels or less.
Run ./data/wider_face_2_voc_add_landmark.py
```Python
 python3 ./data/wider_face_2_voc_add_landmark.py
```
After the program is run and finished, the **wider_face_add_lm_10_10** folder will be generated in the ./data directory. The folder data and data package (1) are the same after decompression. The complete directory structure is as follows:

```Shell
  data/
    retinaface_labels/
      test/
      train/
      val/
    wider_face/
      WIDER_test/
      WIDER_train/
      WIDER_val/
    wider_face_add_lm_10_10/
      Annotations/
      ImageSets/
      JPEGImages/
    wider_face_2_voc_add_landmark.py
```

3. At this point, the VOC training set is ready. There are two scripts: **train_mb_tiny_fd.sh** and **train_mb_tiny_RFB_fd.sh** in the root directory of the project. The former is used to train the **slim version** model, and the latter is used. Training **RFB version** model, the default parameters have been set, if the parameters need to be fine-tuned, please refer to the description of each training parameter in **./train.py**.

4. 运行**train_mb_tiny_fd.sh**和**train_mb_tiny_RFB_fd.sh**即可
```Shell
sh train_mb_tiny_fd.sh 或者 sh train_mb_tiny_RFB_fd.sh
```

## Detect image effects (input resolution: 640x480)
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/26.jpg)
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/2.jpg)
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/4.jpg)
## PS

 - If the actual production scene is medium-distance, large face, and small number of faces, it is recommended to use input size input_size: 320 (320x240) resolution training, and use 320x240 image size input for predictive reasoning, such as using the provided pre-training model. **Mb_Tiny_RFB_FD_train_input_320.pth** for reasoning.
  - If the actual production scene is medium to long distance, small face, and large number of faces, it is recommended to:
 
 （1）Optimal: Input size input_size: 640 (640x480) resolution training, and predictive reasoning with 640x480 image size or larger input size, such as using the provided pre-training model **Mb_Tiny_RFB_FD_train_input_640.pth** for reasoning, higher recall , lower false positives.

 (2) Sub-optimal: input size input_size: 320 (320x240) resolution training, and use 640x480 image size input for predictive reasoning, which is more sensitive to small faces, but false positives will increase.
 
  - The best results for each scene require adjustment of the input resolution to strike a balance between speed and accuracy.
  - Excessive input resolution will enhance the recall rate of small faces, but it will also increase the false positive rate of large and close-range faces, and the speed of reasoning will increase exponentially.
  - Too small input resolution will significantly speed up the reasoning, but it will greatly reduce the recall rate of small faces.
  - The input resolution of the production scene should be as consistent as possible with the input resolution of the model training, and the up and down floating should not be too large.
  
## TODO LIST

  - Join the widerface test code
  - Improve some test data
  - Add MNN, NCNN C++ inference code

##  Reference
 - [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)
 - [libfacedetection](https://github.com/ShiqiYu/libfacedetection/)
 - [RFBNet](https://github.com/ruinmessi/RFBNet)
 - [RFSong-779](https://github.com/songwsx/RFSong-779)
 - [Retinaface](https://github.com/deepinsight/insightface/blob/master/RetinaFace/README.md)

 

