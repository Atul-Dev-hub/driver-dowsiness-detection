## Project: Driver Drowsiness Detection System

### This paper gives links to previous works with their accuracy:
```
    Driver Drowsiness Detection Based on Convolutional Neural Network Architecture Optimization Using Genetic Algorithm
    Link: https://ieeexplore.ieee.org/document/10479511
```

### This paper provides CNN workflow:
```
    Driver Drowsiness Detection using HAAR and CNN
    Link: https://www.ijfmr.com/papers/2025/2/39716.pdf

```

### This is the dataset:
```
    EyeState_Recognition_Dataset
    Link: https://www.kaggle.com/datasets/engkarim1/eyestate-recognition-dataset

    The dataset is divided into two main parts:
        train (used to train your CNN model)
        test (used to evaluate your model)

    Each part has two categories (folders):
        Closed_Eyes: Images with closed eyes.
        Open_Eyes: Images with open eyes.

    Total Images: The dataset contains about 4000 files in total.
        1500 & 1500 : For training of closed & open eyes
        500 & 500 : For testing of closed & open eyes
```

### HAAR CASCADE:
```
    HAAR Cascade is a machine learning-based object detection method.
    In practice, HAAR Cascades are often used in OpenCV for tasks such as face, eye, and mouth detection in live video or static images.
    
    For each detection window, it acts as a binary classifier: It decides if the object (like a nose, face, or eye) is 
    present ("yes") or not ("no") in that region of the image

    Link: https://github.com/opencv/opencv/tree/master/data/haarcascades
```

### openCV:
```
    OpenCV (Open Source Computer Vision Library) is an open-source toolkit widely used for computer vision, image processing, and machine
    learning tasks. It supports real-time operations and has thousands of optimized algorithms for image and video analysis.

    The workflow involves taking an image or video frame and applying the HAAR Cascade classifier's detectMultiScale() function to scan 
    regions of the image at multiple scales and identify rectangles likely to contain the target object (like a face or eye).
```

