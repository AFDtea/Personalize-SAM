import argparse
import os
import sys
import platform
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='model weights path')
    parser.add_argument('--source', type=str, default='data/images', help='source path (image/video/directory)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    return parser.parse_args()

def run_inference(opt):
    # Initialize save directory
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(opt.weights)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.device:
        device = torch.device(opt.device)
    
    # Run inference
    results = model.predict(
        source=opt.source,
        conf=opt.conf_thres,
        iou=opt.iou_thres,
        show=opt.view_img,
        save=True,
        project=opt.project,
        name=opt.name,
        hide_labels=opt.hide_labels,
        hide_conf=opt.hide_conf,
        device=device
    )

    # Save detection results if requested
    if opt.save_txt:
        for i, r in enumerate(results):
            if r.boxes is not None:  # Check if there are any detections
                txt_path = save_dir / 'labels' / f'{Path(r.path).stem}.txt'
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Get boxes, confidences, and class ids
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy()
                
                # Save to file
                with open(txt_path, 'w') as f:
                    for box, conf, cls_id in zip(boxes, confs, cls_ids):
                        # Format: class_id x_center y_center width height confidence
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        x_center = x1 + w/2
                        y_center = y1 + h/2
                        f.write(f'{int(cls_id)} {x_center} {y_center} {w} {h} {conf}\n')

def main(opt):
    run_inference(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)