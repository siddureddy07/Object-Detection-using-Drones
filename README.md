# Object-Detection-using-Drones
Object Detection Models like YOLO, SSD Mobilenet &amp; RetinaNet to detect objects.

# Problem Statement

## Need for the project
Modern drones are equipped with cameras and are very prospective for a variety of commercial uses such as aerial photography, surveillance, etc. In order to massively deploy drones and further reduce their costs, it’s necessary to power drones with smart computer vision and autopilot. In the application of aerial photography, object detection and tracking are essential to capturing key objects in a scene. There are significant challenges with drones due to top-down view angles and real-time constraints. Additionally, The strong weight and area constraint of embedded hardware that limits the drones to run computation intensive algorithms, such as deep learning, with limited hardware resource

Existing Models
Generally, we use Object Detection Models like YOLO, SSD Mobilenet & RetinaNet to detect objects.

When we Directly apply previous models to tackle object detection task on drone-captured scenarios we majorly face three problems.

1.	First, the object scale varies violently because the flight altitude of drones change greatly.
2.	Second, drone-captured images contain objects with high density, which brings in blockage between objects.
3.	Third, drone-captured images always contain confusing geographic elements because of covering large area. These problems make the object detection of drone-captured images very challenging.

## Our Solution
We will train the existing Models using object-detection datasets (VisDrone-2019 Dataset) that are specific to drones so that we can improve the accuracy of these models.

Steps & Workflow:
1.	Change the actual annotations of Vis Drone dataset to Yolov5 label format.
2.	Load existing weights YOLOv5 and fine Tune them.
3.	Train model on the VisDrone-2019 dataset.
4.	Evaluate performance of all models on actual drone footage
5.	Compare the models (YOLOV5,TPH-YOLOV5,SSD,RETINANET) in terms of accuracy, speed and memory.

# About the Dataset

VisDrone is a large-scale benchmark with carefully annotated ground-truth for various important computer vision tasks, to make vision meet drones. The VisDrone2019 dataset is collected by the AISKYEYE team at Lab of Machine Learning and Data Mining, Tianjin University, China. The benchmark dataset consists of 288 video clips formed by 261,908 frames and 10,209 static images, captured by various drone-mounted cameras, covering a wide range of aspects including location (taken from 14 different cities separated by thousands of kilometers in China), environment (urban and country), objects (pedestrian, vehicles, bicycles, etc.), and density (sparse and crowded scenes). Note that, the dataset was collected using various drone platforms (i.e., drones with different models), in different scenarios, and under various weather and lighting conditions. These frames are manually annotated with more than 2.6 million bounding boxes of targets of frequent interests, such as pedestrians, cars, bicycles, and tricycles. Some important attributes including scene visibility, object class and occlusion, are also provided for better data utilization.

Annotation Format

bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion

bbox_left
The x coordinate of the top-left corner of the predicted bounding box

bbox_top
The y coordinate of the top-left corner of the predicted object bounding box

bbox_width
The width in pixels of the predicted object bounding box

bbox_height
The height in pixels of the predicted object bounding box

Score
the score in the DETECTION file indicates the confidence of the predicted bounding box enclosing an object instance.
The score in GROUNDTRUTH file is set to 1 or 0. 
•	1 indicates the bounding box is considered in evaluation.
•	0 indicates the bounding box will be ignored.

# Object Category
The object category indicates the type of annotated object:

•	ignored regions(0)
•	pedestrian(1)
•	people(2)
•	bicycle(3)
•	car(4)
•	van(5)
•	truck(6)
•	tricycle(7)
•	awning-tricycle(8)
•	bus(9)
•	motor(10)
•	others(11)

Truncation
The score in the DETECTION result file should be set to the constant -1.
The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame:

•	no truncation = 0 (truncation ratio 0%)
•	partial truncation = 1 (truncation ratio 1% ~ 50%)

Occlusion
The score in the DETECTION file should be set to the constant -1.
The score in the GROUNDTRUTH file indicates the fraction of objects being occluded

•	no occlusion = 0 (occlusion ratio 0%)
•	partial occlusion = 1 (occlusion ratio 1% ~ 50%)
•	occlusion = 2 (occlusion ratio 50% ~ 100%)
# Exploratory Analysis
•	Ten object categories:
pedestrian, person, car, van, bus, truck, motor, bicycle, awning tricycle, and tricycle

•	Images are manually annotated with more than 2.6 million bounding boxes of targets of frequent interest.

•	288 video clips formed by 261,908 frames and 10,209 static images, captured by various drone-mounted cameras
 
 ![image](https://user-images.githubusercontent.com/90500013/206404488-56bc7351-067a-4eda-a709-a95458cb74ca.png)
![image](https://user-images.githubusercontent.com/90500013/206404529-ffcf9125-8edb-40e5-9383-83743ab05a8e.png)

Sample 1st Image:
Annotations for 1st image:
684,8,273,116,0,0,0,0
406,119,265,70,0,0,0,0
255,22,119,128,0,0,0,0
1,3,209,78,0,0,0,0
708,471,74,33,1,4,0,1

1st Image:
 ![image](https://user-images.githubusercontent.com/90500013/206404598-fae3b7f8-6adb-481b-b2ed-09b775617787.png)


1st Image with some annotations plotted
 ![image](https://user-images.githubusercontent.com/90500013/206404627-46b1dd78-c165-4f76-bd56-ad41561b95d1.png)

## Pre-processing

1.	Firstly, we need to convert the annotations from the custom format to the standard PASCAL VOC format.
2.	For YOLO model, we convert PASCAL VOC to YOLO annotation format.

This prepares our dataset for training the three models.

Refer to pre-processing python notebook for detailed code,

# Objectives

1.	Improve the accuracy of all models by training on drone specific dataset.
2.	Compare accuracy of all the models.
3.	Compare speed of the models.
4.	Compare memory footprint of all the models.

# Evaluation Metrics
The performance of algorithms is evaluated by the average precision (AP) across different object categories and intersection over union (IoU) thresholds. Specifically, AP is computed by averaging over all 10 IoU thresholds, i.e., in the range [0.50 : 0.95] with uniform step size 0.05 of all categories.

## IoU
Intersection over Union (IoU) is used when calculating mAP. It is a number from 0 to 1 that specifies the amount of overlap between the predicted and ground truth bounding box.
•	an IoU of 0 means that there is no overlap between the boxes
•	an IoU of 1 means that the union of the boxes is the same as their overlap indicating that they are completely overlapping
IoU is an important accuracy measure to track when gathering human annotations. The industry best practice is to include a minimum IoU requirement for their human annotation tasks, to ensure that the annotations that are delivered have an IoU >= X (where X = 0.95 is typical) with respect to the “perfect” annotation of that object, as determined by the annotation schema for the project (i.e. box vehicles as tight as possible, including all visible parts of them, including the wheels). State of the are detectors, on the other hand, typically do not perform at a 0.95 IoU
 ![image](https://user-images.githubusercontent.com/90500013/206404820-3aece982-5341-43b4-8a73-21862fc1ba5d.png)

# mAP in short
Mean average precision (mAP) is calculated by first gathering a set of predicted object detections and a set of ground truth object annotations.
•	For each prediction, IoU is computed with respect to each ground truth box in the image.
•	These IoUs are then thresholded to some value (generally between 0.5 and 0.95) and predictions are matched with ground truth boxes using a greedy strategy (i.e. highest IoUs are matched first).
•	A precision-recall (PR) curve is then generated for each object class and the average precision (AP) is computed. A PR curve takes into account the performance of a model with respect to true positives, false positives, and false negatives over a range of confidence values.
•	The mean of the AP of all object classes is the mAP.
•	When evaluating the COCO dataset, this is repeated with 10 IoU thresholds from 0.5 to 0.95 and averaged.


 
# Shortlisted Models & Comparison

Considering all the models and their pros-cons, we select YOLO, SSD & RetinaNet as candidate models for this project.
These models are ‘Single Step Detectors’ and hence are optimized for speed and memory.
We will conduct a comparative study between all three models and evaluate their performance on aerial images dataset – VisDrone.
Pertaining to the experimental results, YOLOv5 achieves 97.70% in terms of mAP@0.5 for all classes, SSD obtains 90.14% mAP in the same term. Meanwhile, regarding recognition speed, YOLOv5 also outperforms SSD.

![image](https://user-images.githubusercontent.com/90500013/206405017-05793832-b003-4e43-b1c0-434948a3b412.png)
![image](https://user-images.githubusercontent.com/90500013/206405256-dd9e6c80-19a3-46d8-b889-0157e89c6067.png)

## Baseline Model

The baseline model for all YOLO, SSD & Retina Net is Convolution Network. 
All the three models use state of the art image feature extractors like VGG-16 and Inception.

# Summary of Models used:

## Model 1: SSD MobileNet V2
•	Trained on: MS COCO dataset.
•	Object Classes: 90 (Which includes our 12 target classes)

This model is used for detection as is, without any additional training or transfer learning.
Hence, this will give us an idea of how models trained on generic object detection datasets like MS COCO perform on Aerial Images.

## Model 2: RetinaNet
•	Trained on: VisDrone Dataset of Aerial Images
•	Object Classes: 12

This model was trained on Google Colab and Local System under time and performance constraints.
Due to these contraints, the model was trained only for 25 epochs, which is not sufficient.
Hence, this will give us an idea of how a self trained model would perform on Aerial Images

## Model 3: TPH YOLOv5
•	Trained on: VisDrone Dataset of Aerial Images
•	Object Classes: 12

This model was trained professionally on the VisDrone dataset with no constraints.
This was model was ranked 5th in the VisDroneDET 2022 challenge.
Hence, this will give us an idea of how a State of the Art model would perform on Aerial Images.


# Comparison of mean Average Precision (mAP)
Mean average precision (mAP) is calculated by first gathering a set of predicted object detections and a set of ground truth object annotations.
•	For each prediction, IoU is computed with respect to each ground truth box in the image.
•	These IoUs are then thresholded to some value (generally between 0.5 and 0.95) and predictions are matched with ground truth boxes using a greedy strategy
•	The mean of the AP of all object classes is the mAP.




# Overall mAP:

Model 1- SSD MobileNet V2: 0.202

Model 2- RetinaNet: 0.2797

Model 3- TPH YOLOv5: 0.408

Actual Annotations:
![image](https://user-images.githubusercontent.com/90500013/206405718-902f96d8-de7e-4bc3-a3ae-cc35b17fb31f.png)

 
Predictions by YOLO: 

![image](https://user-images.githubusercontent.com/90500013/206405788-ff2d7b45-9314-46cc-8388-7ac26170ac5f.png)









Predictions by RetinaNet:
 ![image](https://user-images.githubusercontent.com/90500013/206405756-3ac67705-d246-401e-b01a-6185fc68e181.png)



Predictions by SSD: 

![image](https://user-images.githubusercontent.com/90500013/206405814-510d3107-0327-43b2-a7c0-99a45324545f.png)




