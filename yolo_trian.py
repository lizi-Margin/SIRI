import os
import cv2
from ultralytics import YOLO
from siri.vision.detector import Detector
from siri.global_config import GlobalConfig as cfg


def train(model, dataset_yaml):
    # Train the model
    train_results = model.train(
        data=dataset_yaml,  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device=cfg.device,

        # hsv_h=0.2,
        # hsv_s=0.2
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # print(train_results)
    # print(metrics)

def predict_and_save(model, image_path, output_path):
    image = cv2.imread(image_path)
    results = model.predict(image)
    plotted_image = results[0].plot()
    cv2.imwrite(output_path, plotted_image)
    print(f"Saved prediction result to {output_path}")

if __name__ == '__main__':
    # dataset_name = 'ScrGrabber-tick1-sunone-compat'
    dataset_name = 'ScrGrabber+HMP_IL-sunone-compat'
    model_path = 'model/sunxds_0.7.2.pt'

    root_dir = cfg.root_dir
    model = YOLO(model=model_path)
    dataset = f'{root_dir}/datasets/{dataset_name}/data.yaml'
    train(model, dataset)

    
    image = cv2.imread("./pic_assets/in.jpg")
    image1 = cv2.imread("./pic_assets/in1.jpg")

    predict_and_save(model, "./pic_assets/in.jpg", "./prediction_in.jpg")
    predict_and_save(model, "./pic_assets/in1.jpg", "./prediction_in1.jpg")
