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
