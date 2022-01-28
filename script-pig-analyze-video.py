#!/usr/bin/env python
# coding: utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from PIL import Image
import json
from datetime import datetime
from absl import app, flags
from absl.flags import FLAGS

flags.DEFINE_string('video', './video/pig_yolo_test.mov', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', 'output.avi', 'name of output video')
flags.DEFINE_bool('log_info', True, 'log pig info to a json file')
flags.DEFINE_bool('show_output', False, 'show output video while running')

class PigData:
    def __init__(self, pig_pos, pig_size, tail_type, tail_pos, tail_size, ear_pos, ear_size, ear_visible):
        self.pig_pos = pig_pos
        self.pig_size = pig_size
        self.tail_type = tail_type
        self.tail_size = tail_size
        self.tail_pos = tail_pos
        self.ear_pos = ear_pos
        self.ear_size = ear_size
        self.ear_visible = ear_visible

    def toJson(self):
        return json.dumps({'pig_pos': self.pig_pos, 'pig_size': self.pig_size,
                           'tail_type':self.tail_type, 'tail_size':self.tail_size,
                           'tail_pos':self.tail_pos, 'ear_pos': self.ear_pos, 'ear_size': self.ear_size, 'ear_visible': self.ear_visible
                           }, sort_keys=True, indent=4)

    def __repr__(self):
        return self.toJson()

class Frame:
    def __init__(self, f_number, total_pig, pigs_data):
        self.f_number = f_number
        self.total_pig = total_pig
        self.pigs_data = pigs_data

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def __repr__(self):
        return self.toJson()

class LogData:
    def __init__(self, width, height, time_stamp, frame):
        self.width = width
        self.height = height
        self.time_stamp = time_stamp
        self.frame = frame

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def __repr__(self):
        return self.toJson()

def main(_argv):
    net = cv2.dnn_DetectionModel('yolov4-tiny-custom29.cfg', 'Copy_of_yolov4-tiny-custom29_best.weights')
    net.setInputSize(416, 416)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    tail_ear = cv2.dnn_DetectionModel('tail+ear.cfg', 'tails+ear.weights')
    tail_ear.setInputSize(416, 416)
    tail_ear.setInputScale(1.0 / 255)
    tail_ear.setInputSwapRB(True)

    video = FLAGS.video

    try:
        cap = cv2.VideoCapture(int(video))
    except:
        cap = cv2.VideoCapture(video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(FLAGS.output, fourcc, 20.0, size)

    frame_number = 0

    with open('obj.names', 'rt') as f:
        names = f.read().rstrip('\n').split('\n')

    logData = LogData(int(width), int(height), str(datetime.now()), [])

    while cap.isOpened():
        ret, frame = cap.read()
        frame_number += 1
        counter = 0
        pig_dict = {}

        if ret:
            classes, confidences, boxes = net.detect(frame, confThreshold=0.45, nmsThreshold=0.45)
            pig_index = 0
            display_index = 0

            frame_log = Frame(frame_number,len(boxes), [])

            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                counter = counter + 1
                left, top, width, height = box
                image = frame[top:top + height, left:left + width]
                classes_te, confidences_te, boxes_te = tail_ear.detect(image, confThreshold=0.4, nmsThreshold=0.4)

                tail_na = 'not visible'
                pigData = PigData([int(left), int(top)], [int(width), int(height)], tail_na, [], [], [], [], False)

                for classId_tail_ear, box_tail_ear in zip(classes_te, boxes_te):
                    if classId_tail_ear == 0:  #curly
                        pigData.tail_type = 'curly'
                        left, top, width, height = box_tail_ear
                        pigData.tail_pos = [int(pigData.pig_pos[0]+left), int(pigData.pig_pos[0]+top)]
                        pigData.tail_size = [int(width), int(height)]
                    elif classId_tail_ear == 1:  #straight
                        pigData.tail_type = 'straight'
                        left, top, width, height = box_tail_ear
                        pigData.tail_pos = [int(pigData.pig_pos[0]+left), int(pigData.pig_pos[0]+top)]
                        pigData.tail_size = [int(width), int(height)]
                    elif classId_tail_ear == 2:  #ear
                        pigData.ear_visible = True
                        left, top, width, height = box_tail_ear
                        pigData.ear_pos.append([int(pigData.pig_pos[0]+left), int(pigData.pig_pos[0]+top)])
                        pigData.ear_size.append([int(width), int(height)])

                for cl_te, con_te, box_te in zip(classes_te, confidences_te, boxes_te):
                    label = '%.2f' % confidence
                    box_color = None
                    if cl_te == 0:  #curly
                        label = '(curly): %s' % label
                        box_color = (255, 0, 0)
                    elif cl_te == 1:  #straight
                        label = '(straight): %s' % label
                        box_color = (255, 0, 0)
                    elif cl_te == 2:  #ear
                        label = '(ear): %s' % label
                        box_color = (190, 0, 0)
                    labelsize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    l, t, width, height = box_te
                    left = pigData.pig_pos[0]+l
                    top = pigData.pig_pos[1]+t
                    top = max(top, labelsize[1])
                    cv2.rectangle(frame, [left, top, width, height], color=box_color, thickness=2)
                    cv2.rectangle(frame, (left, top - labelsize[1]), (left + labelsize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (219, 0, 0))

                frame_log.pigs_data.append(pigData)
                logData.frame.append(frame_log)

                pig_dict[pig_index] = pigData
                pig_index = pig_index + 1

            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                label = '%.2f' % confidence
                if display_index in pig_dict.keys():
                    label = '%s(%s tail): %s' % (names[classId], pig_dict[display_index].tail_type, label)
                else:
                    label = '%s(tail not visible): %s' % (names[classId], label)
                labelsize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                left, top, width, height = box
                top = max(top, labelsize[1])
                cv2.rectangle(frame, box, color=(2, 255, 0), thickness=2)
                cv2.rectangle(frame, (left, top - labelsize[1]), (left + labelsize[0], top + baseLine), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                display_index = display_index + 1

            if FLAGS.show_output:
                jsonLog = json.loads(logData.toJson())
                with open(f"{video}.json", "w") as outfile:
                    json.dump(jsonLog, outfile)

            out.write(frame)

            if FLAGS.show_output:
                cv2.namedWindow('Objet Detection YOLO',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Objet Detection YOLO', 1600,900)
                cv2.imshow("Objet Detection YOLO", frame)

            if cv2.waitKey(1) == 13:  #13 is the Enter Key
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
