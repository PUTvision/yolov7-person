
import argparse
import time
from pathlib import Path
from tqdm import tqdm

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def yolo_detect_persons(source, device='gpu') -> dict[str, bool]:
    """
    python3 detect.py --weights yolov7.pt --classes 0 --conf-thres 0.5 --img-size 640 --device cpu  --source
    """
    # source = '/home/przemek/Projects/pp/gnostic-camera-reidentification/src/data_processing/yolov7_person/inference/images'
    save_img = False
    weights = 'yolov7.pt'
    view_img = False
    save_txt = False
    imgsz = 640
    trace = True
    project = 'runs/detect'
    augment = False
    conf_thres = 0.5
    classes = 0
    iou_thres = 0.45
    agnostic_nms = False
    iterdir = False

    results = {}

    # Directories
    save_dir = Path(project)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # if trace:
    #     model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for z, (path, img, im0s, vid_cap) in enumerate(tqdm(dataset, leave=False)):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        results[path] = bool(len(pred[0]) > 0)

        # # Process detections
        # for i, det in enumerate(pred):  # detections per image
        #     p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        #
        #     p = Path(p)  # to Path
        #     if iterdir:
        #         save_path = str(save_dir / str(p.parent).split('/')[-1] / p.name)
        #     else:
        #         save_path = str(save_dir)
        #
        #     Path(save_path).mkdir(exist_ok=True, parents=True)
        #
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        #
        #         # Print results
        #         for c in det[:, -1].unique():
        #             n = (det[:, -1] == c).sum()  # detections per class
        #             s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        #
        #         for k, (*xyxy, conf, cls) in enumerate(reversed(det)):
        #
        #             x1, y1, x2, y2 = xyxy
        #
        #             if conf < conf_thres:
        #                 continue
        #
        #             if (x2 - x1) < 30 or (y2 - y1) < 50:
        #                 continue
        #
        #             # if (x2-x1)*(y2-y1) < 500:
        #             #     continue
        #
        #             # if (x2-x1)/(y2-y1) > 1.5:
        #             #     continue
        #
        #             # if (y2-y1)/(x2-x1) > 4:
        #             #     continue
        #
        #             roi = im0[int(y1):int(y2), int(x1):int(x2)]
        #             cv2.imwrite(f'{save_path}/{z:05d}_{k:03d}.jpg', roi)

    return results

