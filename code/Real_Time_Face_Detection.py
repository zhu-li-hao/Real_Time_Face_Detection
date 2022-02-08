import numpy as np
import cv2 as cv

model_path = 'D:/individual project/OpenCV/Real_Time_Face_Detection/model/yunet.onnx'
faceDetector = cv.FaceDetectorYN.create(model_path, "", input_size=(640, 480))

capture = cv.VideoCapture(0, cv.CAP_DSHOW)
capture.set(3, 640) # 设置摄像头的帧的高为640
capture.set(4, 480) # 设置摄像头的帧的宽为480

while True:
    ret, frame = capture.read()
    faces = faceDetector.detect(frame)
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(frame,(coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (255, 0, 0), thickness=2)
            cv.circle(frame, (coords[4], coords[5]), 2, (255, 0, 0), thickness=2)
            cv.circle(frame, (coords[6], coords[7]), 2, (0, 0, 255), thickness=2)
            cv.circle(frame, (coords[8], coords[9]), 2, (0, 255, 0), thickness=2)
            cv.circle(frame, (coords[10], coords[11]), 2, (255, 0, 255), thickness=2)
            cv.circle(frame, (coords[12], coords[13]), 2, (0, 255, 255), thickness=2)

        cv.imshow('Real_Time_Face_Detection', frame)
        c = cv.waitKey(1)
        # 按esc退出视频
        if c == 27:
            break

    else:
        cv.putText(frame, 'Face is not detected', (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.imshow('Real_Time_Face_Detection', frame)
        c = cv.waitKey(1)
        # 按esc退出视频
        if c == 27:
            break