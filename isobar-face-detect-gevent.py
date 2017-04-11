#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# isobar-face-detect-gevent.py
#
import cv2
import sys
import time
import requests
from subprocess import Popen, PIPE
from ticket import ticket
from imutils.video import VideoStream
from skimage import io
import dlib
import gevent
import signal
from socket import *
from gevent.threadpool import ThreadPool
from gevent.select import select as gselect
import config

#### each steps ####
INIT = 0
DETECT_FACE = 1
PROCESS_FACE = 2
PROCESS_REQUEST = 3
SAVE_FRAME = 4
SHOW_RESULT = 5
END = 6

status = INIT
####################

ENABLE_DLIB = True
ENABLE_FACE_DETECT = True
ENABLE_FPS = False
ENABLE_VIDEO_STREAM = False
DEBUG = True

RT = 0.5
SKIP_FRAME = 5
FRAME_WIDTH = 2560*RT
FRAME_HEIGHT = 1440*RT

# 依不同的 cascade 做調整
# lbpcascade_frontalface: 1.1
# haarcascade_frontalface_alt2: 1.3
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
#MIN_SIZE = 30
MIN_SIZE = 80

postHeader = {}
postHeader['Ocp-Apim-Subscription-Key'] = config.Emotion.key
postHeader['Content-Type'] = 'application/octet-stream'

postVisionHeader = {}
postVisionHeader['Ocp-Apim-Subscription-Key'] = config.Vision.key
postVisionHeader['Content-Type'] = 'application/octet-stream'

gMessage = {}
gResult = {}
startTime = time.time()

if len(sys.argv) < 3:
    print("""
    Usage:
            python isobar-face-detect.py "000123456" "jacky"
    """)
    sys.exit(-1)

userid = sys.argv[1]
username = sys.argv[2]
# cascPath = sys.argv[3]
cascPath = "/Users/isobar/github/nowlab/data/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
detector = dlib.get_frontal_face_detector()

pool = ThreadPool(3)

if ENABLE_VIDEO_STREAM:
    video_capture = VideoStream(usePiCamera=False).start()

else:
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

def dlibFaceDetect(gray):
    global frame
    global status
    global startTime
    faces = detector(gray, 0)
    if len(faces)>0:
        # status = PROCESS_FACE
        startTime = time.time()
        print("dlib Found {0} faces!".format(len(faces)))
    # Drawing a rectangle
    for i, d in enumerate(faces):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
        # cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
        cv2.rectangle(gray, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
    return gray,len(faces)

def faceDetect(gray):
    global frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=(MIN_SIZE, MIN_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces)>0:
        print ("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame

def showText(buf,x,y):
    cv2.putText(frame, buf, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2,(5,5,255),2,cv2.LINE_AA)

def showCenterText(buf):
    rect, baseline = cv2.getTextSize(buf,cv2.FONT_HERSHEY_SIMPLEX,5,10)
    cv2.putText(frame, buf, (int((FRAME_WIDTH-rect[0])*0.5),int(FRAME_HEIGHT*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 5,(255,5,5),10,cv2.LINE_AA)

def processRequest(url, data, headers, params=None):
    print('processRequest')
    result = None
    response = requests.post(url, headers = headers, data = data, params = params)
    if response.status_code == 429:
        if DEBUG:
            print("Message: %s" % (response.json()['error']['message']))
    elif response.status_code == 200 or response.status_code == 201:
        if DEBUG:
            print("Respones:",response.json())
        result = response.json()
    else:
        if DEBUG:
            print("Error code: %d" % (response.status_code))
            try:
                print("Message: %s" % (response.json()['error']['message'] ))
            except Exception as e:
                print(e)
    return result


def emotionAnalysis(frame):
    global status, gResult
    status = PROCESS_REQUEST
    currEmotion = "none"
    _, f = cv2.imencode('.jpg',frame)
    gevent.sleep(3)
    # json = processRequest(config.Emotion.url, f.tobytes(), postHeader)
    # if len(json) > 0:
    #     facerect = json[0]["faceRectangle"]
    #     scores = json[0]["scores"]
    #     currEmotion = max(scores, key=scores.get)
    # print('Analysis: %s' % currEmotion)
    gResult["emotion"] = currEmotion
    status = SAVE_FRAME
    return currEmotion

def visionAnalysis(frame):
    print("visionAnalysis")
    global status, gResult
    status = PROCESS_REQUEST
    age = 0
    gender = "Male"
    description = ""
    _, f = cv2.imencode('.jpg',frame)
    gevent.sleep(3)
    # json = processRequest(config.Vision.url, f.tobytes(), postVisionHeader, {'visualFeatures':'Description,Faces'})
    # if len(json) > 0:
    #     description = json["description"]["captions"][0]["text"]
    #     age = json["faces"][0]["age"]
    #     gender = json["faces"][0]["gender"]
    # print('Age:{}, Gender:{}, Desc:{}'.format(age,gender,description))
    # gResult["description"] = description
    # gResult["age"] = age
    # gResult["gender"] = gender
    gResult["emotion"] = gender
    status = SAVE_FRAME
    return age,gender,description

def start_server():
    print("server starting")
    sock = socket()
    sock.setsockopt(SOL_SOCKET, SO_REUSEADDR,1)
    sock.setblocking(0)
    sock.bind(('',6666))
    sock.listen(5)
    inputs = [sock]
    while True:
        read_sockets,_,_ = gselect(inputs,[],[])
        for s in read_sockets:
            if s is sock:
                client, addr = s.accept()
                print('Connection:{}'.format(addr))
                client.setblocking(0)
                inputs.append(client)
            else:
                data = s.recv(100)
                if len(data) > 0:
                    print('recv:{}'.format(data))
                else:
                    inputs.remove(s)
                    s.close()

def pkill(pname):
    subprocess.call(['pkill', pname])

def main():
    gevent.signal(signal.SIGQUIT, gevent.kill)
    gevent.spawn(main_thread)
    pool.spawn(start_server)
    gevent.wait()

def main_thread():
    global frame, status, startTime, gResult

    cv2.namedWindow("Preview")
    cv2.moveWindow("Preview", 2560-320, 0)

    gevent.sleep(1)
    t = ticket()

    print ("id:{},name:{}".format(userid,username))
    status = DETECT_FACE

    cnt = 0
    file_cnt = 0
    while True:
        # Capture frame-by-frame
        _,frame = video_capture.read()
        frame = cv2.flip(frame,1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(320,180))

        # 顯示文字
        for i in range(1,7):
            showText("Line"+str(i), 50, 100*i)
        showText("{},{}".format(userid,username), 50, 100*(i+1))

        if (status==DETECT_FACE):
            print("臉部偵測")
            if (cnt%SKIP_FRAME==0):
                if ENABLE_FACE_DETECT:
                    if ENABLE_DLIB:
                        resultFrame,faces = dlibFaceDetect(gray)
                    else:
                        resultFrame = faceDetect(gray)
                    cv2.imshow('Preview',resultFrame)
                    if faces > 0:
                        cv2.imwrite("output/{}_{}_{}.png".format(userid,username,file_cnt), frame)
                        file_cnt = file_cnt + 1
                    if file_cnt > 10:
                        status = PROCESS_FACE
        elif (status==PROCESS_FACE):
            # 倒數 5,4,3,2,1,0
            # 5秒後處理圖片
            if time.time()-startTime < 6:
                showCenterText(str(int(time.time()-startTime)))
            if time.time()-startTime > 5:
                # emotionAnalysis(frame)
                # visionAnalysis(frame)
                # pool.spawn(visionAnalysis,frame)
                pool.spawn(emotionAnalysis,frame)
        elif (status==SAVE_FRAME):
            print("存檔")
            # cv2.imwrite("output/save.png", frame)
            status = SHOW_RESULT
            startTime = time.time()
        elif (status==SHOW_RESULT):
            # 顯示結果(5秒)
            if time.time()-startTime < 6:
                showCenterText("r:{},{}".format(gResult["emotion"],str(int(time.time()-startTime))))
            if time.time()-startTime > 5:
                status = END
        elif (status==END):
            pass

        if ENABLE_FPS:
            print("fps:",t.fps())

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cnt = cnt + 1

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("click 'q' to quit")
    main()
