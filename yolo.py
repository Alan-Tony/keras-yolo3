# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

import cv2
from time import sleep
import easyocr
from utils import drawDetection, player_detection, getJerseyColors

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        #"score" : 0.3,
        "score" : 0.85,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes


    def fetch_dict(self, image):

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        output_dict = dict()
        output_dict['pred_classes'] = []
        output_dict['boxes'] = []
        output_dict['scores'] = []
        output_dict['colors'] = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]            

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))

            area_thresh = 0.75
            #Finding area of bbox
            bbox_area = abs(right - left) * abs(bottom - top)
            frame_area = image.size[0] * image.size[1]
            if bbox_area > frame_area * area_thresh**2 or (predicted_class != 'person' and predicted_class != 'sports ball'):
                continue

            #output_dict['pred_classes'].append(predicted_class)
            if(predicted_class == 'person'):
                output_dict['pred_classes'].append("player")
            elif(predicted_class == 'sports ball'):
                output_dict['pred_classes'].append("football")
            output_dict['boxes'].append((left, top, right, bottom))
            output_dict['scores'].append(score)
            output_dict['colors'].append(self.colors[c])

        lengths = [len(x) for x in output_dict.values()]
        assert len(set(lengths)) == 1, 'Unequal lengths of preditcion parameter lists'

        return output_dict

    def detect_image(self, image):

        start = timer()

        output_dict = self.fetch_dict(image)
        #print('Number of detections= {}'.format(len(output_dict['pred_classes'])))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = max(1, (image.size[0] + image.size[1]) // 600)

        result = image.copy()

        for i in range(len(output_dict['pred_classes'])):
            
            text = '{:s}: {:.2f}'.format(output_dict['pred_classes'][i], output_dict['scores'][i])
            result = drawDetection(
                result, 
                output_dict['boxes'][i], text,
                font, thickness)

        end = timer()
        print('Frame predicction: {:.3f} seconds'.format(end - start))

        return result


    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print(video_path)
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    while vid.isOpened():

        return_value, frame = vid.read()

        if cv2.waitKey(1) & 0xFF == ord('q') or return_value is None:
            break

        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        #cv2.imshow('Frame', frame)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        
        cv2.imshow("result", result)
        #print(type(result))
        if isOutput:
            out.write(result)
    
    vid.release()
    cv2.destroyAllWindows()
    yolo.close_session()

def object_track(yolo, video_path, output_path=""):

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"

    while vid.isOpened():

        return_value, frame = vid.read()

        if not return_value:
            break

        cv2.imshow('Press \'q\' to stop', frame)
        sleep(1/video_fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            choice = input('Continue? [y/N]: ')
            if choice in ['n', 'N']:
                break
    cv2.destroyAllWindows()

    if type(frame) != np.ndarray:
        print('Found: {}'.format(type(frame)))
        print('No frame to track from.\nAborting...')
        exit()
    
    #Running YOLOV3 prediction
    image = Image.fromarray(frame)
    output_dict = yolo.fetch_dict(image)

    print('Number of players found = {}'.format(len(output_dict['pred_classes'])))

    tracker_types = ["Boosting", "MIL", "KCF", "TLD", "MedianFlow", "GOTURN", "MOSSE", "CSRT"]
    for i, tracker_type in enumerate(tracker_types):
        print('{}. {:s}'.format(i, tracker_type))

    choice = int(input('Enter index number for tracker type choice: '))
    if choice not in range(len(tracker_types)):
        print('Invalid choice.\nAborting...')
        exit()
    
    func_name = "Tracker" + tracker_types[choice] + "_create"
    trackers = []
    if hasattr(cv2, func_name) :
        #Initialize trackers
        func = getattr(cv2, func_name)
        for i in range(len(output_dict['pred_classes'])):

            box = output_dict['boxes'][i]
            left, top, right, bottom = box
            width = abs(right - left)
            height = abs(bottom - top)
            bbox = (left, top, width, height)

            trackers.append(func())
            trackers[i].init(frame, bbox)

    else:
        print('Specified Object Tracker not found.\nAborting...')
        exit()

    reader = easyocr.Reader(['en'])

    #Tracking detected objects in subsequent frames
    prev_time = timer()
    #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    while vid.isOpened():

        return_value, frame = vid.read()

        if not return_value or cv2.waitKey(1) & 0xFF == ord('q'):
            break

        image = Image.fromarray(frame)
        result = image.copy()

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = max(1, (image.size[0] + image.size[1]) // 600)

        for i in range(len(trackers)):
            
            ok, bbox = trackers[i].update(frame)
            left, top, width, height = bbox

            #Controlling the range of bbox values
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            #Changing the format of bbox to match draw detection's required format
            bbox = (left, top, left + width, top + height)
            #Get jersey color
            jersey_color = getJerseyColors(frame[top : top + height, left : left + width, :], bbox)
            if ok:
                #text = '{:s}: {:.2f}'.format(output_dict['pred_classes'][i], output_dict['scores'][i])
                #result = drawDetection(result, bbox, text, font, thickness)
                result = player_detection(result, image, bbox, reader, font, thickness, text_box_fill=jersey_color)
        result = np.asarray(result)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        
        cv2.imshow("result", result)

        if isOutput:
            out.write(result)

    vid.release()
    cv2.destroyAllWindows()
    yolo.close_session()