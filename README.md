# Clip Localization with YOLO, FasterRCNN Resnet50 and Resnet18

This project was an exam of the module "AI project" as part of our studies in AI.
In cooperation with the company RIWOlink, we had to come up with a solution for clip localization.

In a laparoscopic procedure, the camera is guided into the abdominal cavity via a shaft, in which minimally invasive surgery 
is then performed. One of the most common procedures is cholecystectomy, the removal of the gallbladder. 
In order to be able to safely remove the gallbladder, an artery (arteria cystica) and a bile duct (ductus cysticus) 
must be securely blocked. Mechanical clips made of metal are used for these ligations and placed on the tissue with a 
special instrument (clipper / clip applicator).

An AI application that contributes to patient safety could recognize whether the clips were placed in the appropriate place.
Each clip should be marked over a rectangle in the image.

## Install
To install all the necessary packages, you need to navigate to the root directory and run the following command in the terminal:

```sh
$ pip install -r requirements.txt
```

## Run Inference
To start a container with an inference of the trained YOLO and Faster-RCNN Resnet50 model you have to run the following command in the terminal:

```sh
$ docker-compose up
```

The web server runs on [http://localhost:8080](http://localhost:8080).
If you want to run the web interface on your local machine you have to run the _main.py_ file and connect to [http://localhost:8000](http://localhost:8000).


## Project Structure
The images and annotations must be stored in YOLO format in the _clip_ folder. 

In addition, the models that are to be available for selection for the inference must be inserted in the _src/models_ folder. 
These can then be selected on the web interface. There is one folder for YOLO and one for Faster-RCNN.

The images to be used for the inference must be placed in the _src/test_images_ folder. 

The whole folder structure is shown below:

```sh
├── Dockerfile
├── README.md
├── docker-compose.yml
├── documentation
├── requirements.txt
├── requirements_docker.txt
├── dataset
│   ├── clip
│   ├── data.yaml
│   └── val_images
└── src
    ├── main.py
    ├── evaluation
    │   ├── bounding_box.py
    │   ├── converter.py
    │   ├── detection_export_resnet.py
    │   ├── detection_export_yolo.py
    │   ├── enumerators.py
    │   ├── evaluator.py
    │   ├── run.py
    │   └── validations.py
    ├── models
    │   ├── faster_rccn
    │   └── yolo
    ├── preprocessing
    │   ├── adjust_distribution.py
    │   ├── convert_yolo_to_pascalvoc.py
    │   ├── data_augmentation.py
    │   ├── readjust_annotations.py
    │   ├── remove.py
    │   └── rename.py
    ├── resnet
    │   ├── detect_fasterrcnn.py
    │   ├── detect_resnet.py
    │   ├── train_fasterrcnn.py
    │   ├── train_resnet.py
    │   └── models
    ├── static
    │   ├── style.css
    │   └── style_yolo.css
    ├── templates
    │   ├── index.html
    │   └── index_yolo.html
    ├── test_images
    ├── utils
    │   ├── coco.py
    │   ├── evaluation_utils.py
    │   ├── fasterrcnn_utils.py
    │   ├── general_utils.py
    │   ├── image_handler.py
    │   ├── logging.py
    │   ├── path.py
    │   ├── train_utils.py
    │   └── transforms.py
    └── yolov8
        ├── detect.py
        └── train.py
```

## Contributions
Initially, the data sets had to be labeled. For this purpose, the data sets were divided among the team members.
Furthermore, the documentation for the individual progress presentations was done in cooperation.
In the further course of the project, the tasks were divided as follows:

_Simon Wolf_ 
- Export the predicted bounding boxes
- Adjustment of the distribution
- Convert yolo to pascal
- Resnet18
- Yolo detection script
- Responsible for training different models including hyperparameter tuning


_Tim Staudinger_
- Calculation of the mAP according to Pascal Voc implementation
- Data Augmentation
- Readjust annotations
- Faster-RCNN
- Run application inside a container
- Web interface
- Yolo trainings script