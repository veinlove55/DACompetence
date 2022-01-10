import os
import sys
import numpy as npp
import pandas as pd
import cv2
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pickle
from easydict import EasyDict as edict
from getopt import getopt
import math
import torch
import time
from PINet.parameters import *
from PINet.data_loader import Generator
import PINet.agent as agent
from copy import deepcopy
import PINet.util as util
from yolov5.utils.torch_utils import *
from yolov5.models.experimental import *
from yolov5.utils.general import *
from yolov5.utils.datasets import *
from PINet.test import test
from yolov5.ObjectTracking import *

p = Parameters()
root_path = os.getcwd()
sys.path.insert(0, root_path + '/yolov5')
sys.path.insert(0, root_path + '/PINet')


def load_video_lane_model():
    if p.model_path == "":
        raise ValueError('Please provide video lane model')
    else:
        lane_agent = agent.Agent()
        lane_agent.load_weights(root_path)

    if torch.cuda.is_available():
        lane_agent.cuda()

    lane_agent.evaluate_mode()

    return lane_agent


def restore_points(img1_shape, out_x, out_y, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gainy = img1_shape[0] / img0_shape[0]
        gainx = img1_shape[1] / img0_shape[1]
        pad = (img1_shape[1] - img0_shape[1] * gainx) / 2, (img1_shape[0] - img0_shape[0] * gainy) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    out_ximg = []
    out_yimg = []
    if len(out_x) > 1 or len(out_y) > 1:
        raise ValueError('Video lane point is not one length')

    for out_xx in out_x[0]:
        out_xx = torch.from_numpy(np.array(out_xx)).float()
        out_xx -= pad[0]
        out_xx /= gainx
        out_xx.clamp_(0, img0_shape[1])
        out_xx = out_xx.round()
        out_ximg.append(out_xx)
    for out_yy in out_y[0]:
        out_yy = torch.from_numpy(np.array(out_yy)).float()
        out_yy -= pad[1]
        out_yy /= gainy
        out_yy.clamp_(0, img0_shape[0])
        out_yy = out_yy.round()
        out_yimg.append(out_yy)

    return out_ximg, out_yimg


def video_lane_detection(lane_agent, frame, img_w, img_h):
    frame = cv2.resize(frame, (512, 256)) / 255.0
    frame = np.rollaxis(frame, axis=2, start=0)
    out_x, out_y = test(lane_agent, np.array([frame]))
    # ti[0] = cv2.resize(ti[0], (img_w, img_h))
    out_x, out_y = restore_points([256, 512], out_x, out_y, [img_h, img_w])

    return out_x, out_y


def warning_on_current_lane_mark(x_line, lss_switch, image):
    warning_lss = False
    color_line = (50, 215, 14)

    # calculate the warning zone
    point_x_middle = int(image.shape[1]/2)
    x_left_width = int(point_x_middle/2)
    x_right_width = x_left_width
    x_left_threshold = point_x_middle - x_left_width
    x_right_threshold = point_x_middle + x_right_width

    # x_line position of last point of i
    x_first_point = int(x_line[0])
    left_on = x_first_point >= x_left_threshold
    right_on = left_on and x_first_point <= x_right_threshold
    lss_on = right_on and lss_switch
    if lss_on:
        color_line = (0, 0, 255)
        warning_lss = True
    # if ego car was on the current line, the color of line will be red, otherwise, it will be green
    return color_line, warning_lss


def select_lon_target_object(point_x, point_y, boxes, distance):
    successful_select = False
    object_index = 0

    # get the ego line of left and right
    if len(point_x) >= 3:
        x_left_line = point_x[1]
        y_left_line = point_y[1]
        x_right_line = point_x[2]
        y_right_line = point_y[2]
    else:
        return successful_select, object_index

    min_length = min(len(x_left_line), len(x_right_line))
    if min_length < 8:
        return successful_select, object_index

    # get two points from left and right line
    x1_left = float(x_left_line[3])
    y1_left = float(y_left_line[3])
    x2_left = float(x_left_line[8])
    y2_left = float(y_left_line[8])

    x1_right = float(x_right_line[3])
    y1_right = float(y_right_line[3])
    x2_right = float(x_right_line[8])
    y2_right = float(y_right_line[8])

    center_of_line_x = (x1_left + x1_right) / 2
    # calculate the a and b of left and right line
    a_left = (y2_left - y1_left) / (x2_left - x1_left)
    b_left = y1_left - a_left * x1_left

    a_right = (y2_right - y1_right) / (x2_right - x1_right)
    b_right = y1_right - a_right * x1_right

    max_distance = 1000
    max_difference = 1000
    # loop all boxes and check if one of them was the target object
    for i in range(len(boxes)):
        object_in_ego_line = False
        box = boxes[i]
        rear_left_box = np.array((int(box[0]), int(box[3])))
        rear_right_box = np.array((int(box[2]), int(box[3])))
        center_of_object = (rear_left_box[0] + rear_right_box[0]) / 2
        rear_left_box[0] = rear_left_box[0] + rear_left_box[0] / 3
        rear_right_box[0] = rear_right_box[0] - rear_right_box[0] / 3

        difference_object_line = abs(center_of_object - center_of_line_x)
        is_rear_left_in_range = False
        is_rear_right_in_range = False
        left_point_upon_left_line = rear_left_box[1] > a_left * rear_left_box[0] + b_left
        left_point_upon_right_line = rear_left_box[1] > a_right * rear_left_box[0] + b_right
        right_point_upon_left_line = rear_right_box[1] > a_left * rear_right_box[0] + b_left
        right_point_upon_right_line = rear_right_box[1] > a_right * rear_right_box[0] + b_right

        if left_point_upon_left_line: #and left_point_upon_right_line:
            is_rear_left_in_range = True
        if right_point_upon_left_line and right_point_upon_right_line:
            is_rear_right_in_range = True

        if is_rear_left_in_range and center_of_object < 640 and distance[i] < 30:#or is_rear_right_in_range:
            object_in_ego_line = True
        if object_in_ego_line and distance[i] < max_distance:
            object_index = i
            max_distance = distance[i]
            successful_select = True
            max_difference = difference_object_line
    """
    if min_length == 0:
        return successful_select, object_index

    x_left_line = x_left_line[:min_length]
    y_left_line =y_left_line[:min_length]
    x_right_line = x_right_line[:min_length]
    y_right_line = y_right_line[:min_length]

    # loop the boxes
    for box in boxes:
        index = 0
        # get the width and rear left rear right point of the box
        box_width = float(box[2] - box[0])
        rear_left_box = np.array((int(box[0]), int(box[3])))
        rear_right_box = np.array((int(box[2]), int(box[3])))

        # init segment index of object belong to
        segment = 0
        is_in_ego_lane = False
        max_distance = 1000
        weight_in_ego_lane = 0
        for jj in range(len(y_left_line)-2):
            if int(rear_left_box[1]) < int(y_left_line[jj]) and int(rear_left_box[1]) >= int(y_left_line[jj+1]):
                segment = jj
                break
        if segment == 0:
            return successful_select, index
        # check if the object was in ego lane, check if one of the point of box was in jj
        # if the rear left point was in range of jj
        left_in_range = False
        right_in_range = False
        if rear_left_box[0] > int(x_left_line[segment]) and rear_left_box[0] < x_right_line[segment]:
            left_in_range = True

        if rear_right_box[0] > int(x_left_line[segment]) and int(rear_right_box[0] < x_right_line[segment]):
            right_in_range = True

        # calculate weight object in ego lane
        if left_in_range and right_in_range:
            weight_in_ego_lane = 1
        elif left_in_range:
            weight_in_ego_lane = (int(x_right_line[segment]) - rear_right_box[0]) / box_width
        elif right_in_range:
            weight_in_ego_lane = (rear_left_box[0] - int(x_left_line[segment])) / box_width
        else:
            weight_in_ego_lane = 0
        if distance[index] < max_distance and weight_in_ego_lane > 0.5:
            max_distance = distance[index]
            successful_select = True
            object_index = index
        index = index + 1
    """
    return successful_select, object_index


def video_lane_draw_points(point_x, point_y, lss_switch, img):
    color_index = 0

    for i, j in zip(point_x, point_y):
        color_line = (50, 215, 14)
        warning_lss = False
        color_index += 1
        if color_index > 12:
            color_index = 12
        if color_index == 2 or color_index == 3:
            color_line, warning_lss = warning_on_current_lane_mark(i, lss_switch, img)
            show_number = len(i)
        for index in range(len(i)):
            cv2.circle(img, (int(i[index]), int(j[index])), 2, color_line, -1)


def load_video_object_model():
    device = select_device()
    half = device.type != 'cpu'
    model = attempt_load('./yolov5/yolov5s.pt', map_location=device)  # load FP32 model
    imgsz = check_img_size(640, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    return model, imgsz, device


def video_object_detection(frame, model, imgsz, device, objFactory, is_first_cycle):
    half = device.type != 'cpu'
    img0 = frame

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Padded resize
    img = letterbox(frame, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0labels.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.5, 0.45, None, agnostic=False)
    t2 = time_synchronized()

    labels = []
    boxes = []
    clses = []
    distance = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s = ""
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            j = len(det)-1
            # Store results
            for *xyxy, conf, cls in reversed(det):
                c_ratio = 400
                n_y = int(xyxy[3]) - int(xyxy[1])
                if int(cls) == 0:
                    h_obj = 1.8
                elif int(cls) == 2:
                    h_obj = 2
                elif int(cls) == 7:
                    h_obj = 3
                else:
                    h_obj = 0

                distance_obj = (c_ratio * h_obj) / n_y
                """*************GuanTing****************Tracking"""
                #obj2draw = objFactory.AssociationAndUpdate(xyxy, distance_obj, is_first_cycle)
                """*************GuanTing****************Tracking"""
                label = '%s %s %s %.2f %0.2f %s' % ('handle', j, names[int(cls)], conf, distance_obj, 'm')
                j = j - 1
                labels.append(label)
                boxes.append(xyxy)
                distance.append(distance_obj)
                clses.append(cls)

    return labels, boxes, clses, colors, distance


def gstreamer_pipeline(
    capture_width=640,
    capture_height=480,
    fps=60,
    display_width=640,
    display_height=480,
    flip_method=0,
):
    ret1 = 'nvarguscamerasrc ! '
    ret2 = 'video/x-raw(memory:NVMM), width=(int){}, height=(int){}, '.format(capture_width, capture_height)
    ret3 = 'format=(string)NV12, framerate=(fraction){}/1 ! '.format(fps)
    ret4 = 'nvvidconv flip-method={} ! '.format(flip_method)
    ret5 = 'video/x-raw, width=(int){}, height=(int){}, '.format(display_width, display_height)
    ret6 = 'format=(string)BGRx ! '
    ret7 = 'videoconvert ! '
    ret8 = 'video/x-raw, format=(string)BGR ! appsink'

    return ret1 + ret2 + ret3 + ret4 + ret5 + ret6 + ret7 + ret8


def main(argv):
    # -----------------------------------------
    # parse arguments
    # -----------------------------------------
    opts, args = getopt(argv, '', ['inputPath=', 'outputPath=', 'mode='])

    # defaults
    input_path = None
    output_path = None
    mode = None

    # read opts
    for opt, arg in opts:
        if opt in ('--inputPath'): input_path = arg
        if opt in ('--outputPath'): output_path = arg
        if opt in ('--mode'): mode = int(arg)

    # required opt
    if input_path is None or not os.path.exists(input_path):
        raise ValueError('Please provide inputPath, e.g., --inputPath')
    if output_path is None:
        raise ValueError('Please provide outputPath, e.g., --outputPath')
    if mode is None or mode not in [0, 1]:
        raise ValueError('Please provide mode, e.g., --mode 0 (0: video file, 1: camera stream)')

    # Handle video input
    if mode == 0:
        cap = cv2.VideoCapture(input_path)
    elif mode == 1:
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    else:
        raise ValueError('Please provide right mode, e.g., --mode 0 (0: video file, 1: camera stream)')

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Make the code deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Handle video output
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter(output_path, fourcc, 20, (img_w, img_h))

    # Start load video lane model
    lane_model = load_video_lane_model()
    # End

    # Start load video object model
    object_model, imgsz, device = load_video_object_model()
    # End
    # lateral and longitudinal function active
    lss_switch = False
    lon_switch = False
    is_first_cycle = True
    """*************GuanTing****************Tracking"""
    objFactory = objectsFactory()
    """*************GuanTing****************Tracking"""
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        torch.cuda.synchronize()
        prevTime = time_synchronized()
        with torch.no_grad():
            # Infer video lane
            point_x, point_y = video_lane_detection(lane_model, frame, img_w, img_h)
            # Infer video object
            labels, boxes, clses, colors, distance = video_object_detection(frame, object_model, imgsz, device, objFactory, is_first_cycle)
            distance.sort()
        is_first_cycle = False
        curTime = time_synchronized()
        sec = curTime - prevTime
        fps = 1 / (sec)
        s = "FPS : " + str(fps)

        # select target object whs6szh
        is_target_object_selected = False
        object_index = None
        if lon_switch:
            is_target_object_selected, object_index = select_lon_target_object(point_x, point_y, boxes, distance)

        # Draw video points in frame
        video_lane_draw_points(point_x, point_y, lss_switch, frame)
        object_color = (255, 0, 0)
        # Draw video objects in frame
        for index, _ in enumerate(labels):
            if is_target_object_selected and object_index == index:
                object_color = (0, 0, 255)
            plot_one_box(boxes[index], frame, label=labels[index], color=object_color, line_thickness=3)
            object_color = (255, 0, 0)

        # put object pool of current cycle to last cycle list and clean up the current object pool
        objFactory.objectsPoolLastCycle = objFactory.objectsPool
        objFactory.objectsPool = [None] * objFactory.Param_ObjectsNumber

        # cv2.putText(frame, s, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.imshow('frame', frame)
        vout.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    vout.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
