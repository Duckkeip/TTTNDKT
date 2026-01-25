from ultralytics import YOLO
import streamlit as st

class Detector:
    def __init__(self, plate_model_path, sv_model_path):
        self.plate_model = YOLO(plate_model_path)
        self.sv_model = YOLO(sv_model_path)

    def detect_plate(self, frame, conf=0.5):
        return self.plate_model.predict(frame, conf=conf, imgsz=416)[0]

    def detect_sv(self, frame, conf=0.5):
        return self.sv_model.predict(frame, conf=conf)[0]