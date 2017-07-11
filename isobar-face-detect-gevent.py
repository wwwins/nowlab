#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# isobar-face-detect-gevent.py
#
import cv2
import sys
import time
import datetime
import requests
import random
import string
import subprocess
import math
import logging
from ticket import ticket
from imutils.video import VideoStream
from skimage import io
import dlib
import gevent
import signal
import numpy as np
from socket import *
from gevent.threadpool import ThreadPool
from gevent.server import StreamServer
from PIL import Image, ImageDraw, ImageFont
import config
import siri

#### each steps ####
INIT = 0
WAITING = 1
CHECKIN = 2
SIRI_TIME = 3
DETECT_FACE = 4
PROCESS_FACE = 5
PROCESS_REQUEST = 6
PROCESSING = 7
SAVE_FRAME = 8
SHOW_RESULT = 9
END = 10

status_text = ['init','waiting','checkin','siri time','detect face','process face','process request','processing','save','result','end']
status = INIT
####################

HOST = "10.65.136.40"
ENABLE_DLIB = True
ENABLE_FACE_DETECT = True
ENABLE_FPS = False
ENABLE_VIDEO_STREAM = False
# Vision or Face
# API_TYPE = 'Vision'
API_TYPE = 'Face'
DEBUG = False

RT = 0.5
SKIP_FRAME = 1
FRAME_WIDTH = 2560*RT
FRAME_HEIGHT = 1440*RT

# 依不同的 cascade 做調整
# lbpcascade_frontalface: 1.1
# haarcascade_frontalface_alt2: 1.3
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
# MIN_SIZE = 30
MIN_SIZE = 80

postHeader = {}
postHeader['Ocp-Apim-Subscription-Key'] = config.Emotion.key
postHeader['Content-Type'] = 'application/octet-stream'

postVisionHeader = {}
postVisionHeader['Ocp-Apim-Subscription-Key'] = config.Vision.key
postVisionHeader['Content-Type'] = 'application/octet-stream'

postFaceHeader = {}
postFaceHeader['Ocp-Apim-Subscription-Key'] = config.Face.key
postFaceHeader['Content-Type'] = 'application/octet-stream'

gResult = {}
startTime = time.time()

# if len(sys.argv) < 3:
#     print("""
#     Usage:
#             python isobar-face-detect.py "000123456" "jacky"
#     """)
#     sys.exit(-1)

# userid = sys.argv[1]
# username = sys.argv[2]
# cascPath = sys.argv[3]
cascPath = "/Users/isobar/github/nowlab/data/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
detector = dlib.get_frontal_face_detector()

pool = ThreadPool(4)

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
    # Dlib's face detector is trained to process 80x80 faces
    # 如果要進行 40x40 臉部偵測，需要將圖放大
    # http://dlib.net/face_detection_ex.cpp.html
    gray = cv2.pyrUp(gray)
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
    return gray, faces

def faceDetect(gray):
    global frame,status,startTime
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=(MIN_SIZE, MIN_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces)>0:
        startTime = time.time()
        print ("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return gray, faces

def crop(img, faces):
    for d in faces:
        # x, y, x+w, y+h -> d.left(), d.top(), d.right(), d.bottom()
        # HxW
        return img[d.top():d.bottom(), d.left():d.right()]

def showText(buf,x,y):
    cv2.putText(frame, buf, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(5,5,200),2,cv2.LINE_AA)

def showCenterText(buf):
    rect, baseline = cv2.getTextSize(buf,cv2.FONT_HERSHEY_SIMPLEX,5,10)
    cv2.putText(frame, buf, (int((FRAME_WIDTH-rect[0])*0.5),int(FRAME_HEIGHT*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 5,(5,5,200),10,cv2.LINE_AA)

def showCenterBottomText(buf):
    rect, baseline = cv2.getTextSize(buf,cv2.FONT_HERSHEY_SIMPLEX,5,10)
    cv2.putText(frame, buf, (int((FRAME_WIDTH-rect[0])*0.5),int(FRAME_HEIGHT-rect[1])), cv2.FONT_HERSHEY_SIMPLEX, 5,(5,5,200),10,cv2.LINE_AA)

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
    if (status==PROCESS_REQUEST):
        status = SAVE_FRAME
    return currEmotion

def visionAnalysis(frame):
    print("visionAnalysis")
    global status, gResult
    status = PROCESS_REQUEST
    age = random.randint(1,100)
    gender = random.choice(["male","female"])
    description = ""
    _, f = cv2.imencode('.jpg',frame)
    if DEBUG:
        gevent.sleep(3)
        # gender = random.choice(["Male","Female"])
        # age = random.randint(1,100)
        gResult["emotion"] = "neutral"
    else:
        json = processRequest(config.Vision.url, f.tobytes(), postVisionHeader, {'visualFeatures':'Description,Faces'})
        if len(json) > 0:
            description = json["description"]["captions"][0]["text"]
            if len(json["faces"]) > 0:
                age = json["faces"][0]["age"]
                gender = json["faces"][0]["gender"]
    print('Age:{}, Gender:{}, Desc:{}'.format(age,gender,description))
    gResult["description"] = description
    gResult["age"] = age
    gResult["gender"] = gender.lower()
    if (status==PROCESS_REQUEST):
        status = SAVE_FRAME
    return age,gender,description

def faceAnalysis(frame):
    print("faceAnalysis")
    global status, gResult
    status = PROCESS_REQUEST
    age = random.randint(1,100)
    gender = random.choice(["male","female"])
    description = ""
    _, f = cv2.imencode('.jpg',frame)
    if DEBUG:
        gevent.sleep(3)
        # gender = random.choice(["Male","Female"])
        # age = random.randint(1,100)
        gResult["emotion"] = "neutral"
    else:
        json = processRequest(config.Face.url, f.tobytes(), postFaceHeader, {'returnFaceAttributes':'age,gender'})
        print("json:", json)
        if len(json) > 0:
            description = json[0]["faceId"]
            if len(json[0]["faceAttributes"]) > 0:
                age = json[0]["faceAttributes"]["age"]
                gender = json[0]["faceAttributes"]["gender"]
    print('Age:{}, Gender:{}, Desc:{}'.format(age,gender,description))
    gResult["description"] = description
    gResult["age"] = age
    gResult["gender"] = gender.lower()
    if (status==PROCESS_REQUEST):
        status = SAVE_FRAME
    return age,gender,description


'''
    userid = '1234567890'
    userid = 'U7bf88313277b1c4e40200dbd7f6af8a8 Ammon'
'''
def get_user_info(userid):
    global status, gResult
    status = PROCESS_REQUEST
    buf = userid.split(' ')
    if (len(buf)>1):
        userid = buf[0]
        gResult['username'] = buf[1]
        gResult['time'] = str(datetime.datetime.now().hour) + ':00'
        gResult['email'] = buf[1]
        status = SIRI_TIME
    else:
        # gevent.sleep(3)
        # gResult["username"] = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        # return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        # response = requests.post("https://uinames.com/api/", {"userid":userid})
        response = requests.get("http://{}:16888/api/checkInByRFID?dev=0&rfid={}".format(HOST, str(userid)))
        try:
            data = response.json()
        except Exception as e:
            data = {'name': ''}
            print(e)
        if data.get('user'):
            # gResult['username'] = data['name'].encode('utf-8')
            gResult['username'] = data['user']['name'].encode('utf-8')
            gResult['time'] = data['user']['time']
            gResult['email'] = data['user']['email'].split('@')[0]
            print("name:{}".format(gResult['username']))
            if (status==PROCESS_REQUEST):
                status = SIRI_TIME
        else:
            print("key error")
            status = WAITING

def handle(socket, address):

    global status
    print('New connection from:{}'.format(address))
    rfileobj = socket.makefile(mode='rb')
    while True:
        line = rfileobj.readline()
        if not line:
            print("client disconnected")
            break
        if (status==END or status==WAITING):
            gResult["userid"] = line.rstrip()
            status = CHECKIN
            print("echoed %r" % line)
    rfileobj.close()


def sayit(contents):
    subprocess.Popen(['say', contents])

def say_welcome(contents):
    global status
    status = PROCESSING
    sayit(contents)
    gevent.sleep(5)
    sayit(random.choice(siri.SIRI_BOSS))
    gevent.sleep(3)
    sayit(random.choice(siri.SIRI_123))
    gevent.sleep(2)
    if (status==PROCESSING):
        status = DETECT_FACE

def say_result(contents):
    global status
    status = PROCESSING
    sayit(contents)
    gevent.sleep(6)
    if (status==PROCESSING):
        status = SHOW_RESULT

def say_bye(contents):
    global status
    status = PROCESSING
    sayit(contents)
    gevent.sleep(4)
    if (status==PROCESSING):
        status = END

def get_gradient_image(image):
    imgsize = image.size
    innerColor = [0, 92, 151]
    outerColor = [54, 55, 149]
    for y in range(imgsize[1]):
        for x in range(imgsize[0]):

            #Find the distance to the center
            distanceToCenter = math.sqrt((x - imgsize[0]/2) ** 2 + (y - imgsize[1]/2) ** 2)
            #Make it on a scale from 0 to 1
            distanceToCenter = float(distanceToCenter) / (math.sqrt(2) * imgsize[0]/2)
            #Calculate r, g, and b values
            r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)
            g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)
            b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)

            #Place the pixel
            image.putpixel((x, y), (int(r), int(g), int(b)))
    return image

def get_image_text(contents):

    image = Image.new("RGB",(600,500),(0,0,0))

    image = get_gradient_image(image)
    draw = ImageDraw.Draw(image)
    text = contents

    font = ImageFont.truetype("/Library/Fonts/PingFang.ttc", 72)
    # draw watermark in the bottom right corner
    draw.text((50,50),unicode(text, 'UTF-8'), font=font, spacing=20, fill=(240,240,240))

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def show_image_text(contents):
    frame_title = get_image_text(contents)
    cv2.imshow("Title",frame_title)

def add_overlay_circle(frame, alpha):
    overlay = frame.copy()
    output = frame.copy()
    cv2.circle(overlay,(640,230),150,(255,255,255),4)
    cv2.line(overlay, (520,320),(320,700),(255,255,255),4)
    cv2.line(overlay, (760,320),(960,700),(255,255,255),4)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)
    return output

def pkill(pname):
    subprocess.call(['pkill', pname])

def init():
    global status,gResult,cnt,file_cnt
    status = WAITING
    gResult["userid"] = ''
    gResult["username"] = ''
    gResult["time"] = ''
    gResult["email"] = ''
    cnt = 0
    file_cnt = 0

def main():
    gevent.signal(signal.SIGQUIT, gevent.kill)
    gevent.spawn(main_thread)

    server = StreamServer(('', 5555), handle)
    server.start()
    print("Starting ISOBAR facial recognition server")

    gevent.wait()


def main_thread():
    global frame, status, startTime, gResult,cnt,file_cnt

    cv2.namedWindow("Video")
    cv2.moveWindow("Video", 690, 0+150)

    cv2.namedWindow("Preview")
    # cv2.moveWindow("Preview", 2560-320-100, 0)
    cv2.moveWindow("Preview", 690+320, 750+150)

    # cv2.namedWindow("SubBackground")
    # cv2.moveWindow("SubBackground", 690+1280, 0)

    cv2.namedWindow("Title")
    cv2.moveWindow("Title", 80, 0+150)

    alpha = 0.6
    gevent.sleep(1)
    t = ticket()
    # fixed OpenCV Error
    # https://github.com/opencv/opencv/issues/6055#issuecomment-200354222
    cv2.ocl.setUseOpenCL(False)
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg = cv2.createBackgroundSubtractorKNN(500,400,True)
    init()

    while True:
        # Capture frame-by-frame
        _,frame = video_capture.read()
        frame = cv2.flip(frame,1)

        gray = cv2.resize(frame,(320,180))
        mask = fgbg.apply(gray)
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)
        result = cv2.bitwise_and(gray,gray,mask=mask)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('SubBackground',result)

        # 顯示文字
        # for i in range(1,7):
        #     showText("Line"+str(i), 50, 100*i)
        # showText("Waiting...", 50, 100*(i+1))
        if DEBUG:
            showText("status:{}".format(status_text[status]), 50,50)

        userid = gResult["userid"]
        username = gResult["username"]

        if (status==WAITING):
            """
            if DEBUG:
                print("Waiting")
            """
        elif (status==CHECKIN):
            gResult["username"] = ''
            cnt = 0
            file_cnt = 0
            # 依讀卡機傳來卡號，取得使用者資料
            pool.spawn(get_user_info,userid)
        elif (status==SIRI_TIME):
            buf = '{} {}, 歡迎來到, 安索帕 體驗會'.format(random.choice(siri.SIRI_WELCOME), gResult['username'])
            pool.spawn(say_welcome, buf)
            pool.spawn(show_image_text, "場次: {}\n姓名: {}\n帳號: {}".format(gResult['time'],gResult['username'],gResult['email']))
        elif (status==DETECT_FACE):
            print("臉部偵測")
            frame_with_circle = add_overlay_circle(frame,alpha)
            if (cnt%SKIP_FRAME==0):
                if ENABLE_FACE_DETECT:
                    if ENABLE_DLIB:
                        resultFrame,faces = dlibFaceDetect(gray)
                    else:
                        resultFrame,faces = faceDetect(gray)
                    cv2.imshow('Preview',resultFrame)
                    if len(faces) > 0:
                        cv2.imwrite("output/{}_{}_{}.png".format(userid,username,file_cnt), frame)
                        file_cnt = file_cnt + 1
                    if file_cnt > 10:
                        cropFrame = crop(resultFrame,faces)
                        cv2.imwrite("output/{}_{}_crop.png".format(userid,username), cropFrame)
                        status = PROCESS_FACE
        elif (status==PROCESS_FACE):
            # 倒數 5,4,3,2,1,0
            # 1秒後處理圖片
            if time.time()-startTime < 2:
                # showCenterText(str(int(time.time()-startTime)))
                showCenterBottomText("processing....")
            if time.time()-startTime > 1:
                # emotionAnalysis(frame)
                # visionAnalysis(frame)
                sayit('處理中')
                cropFrame = cv2.imread("output/{}_{}_crop.png".format(userid,username))
                if (API_TYPE=="Vision"):
                    pool.spawn(visionAnalysis,cropFrame)
                else:
                    pool.spawn(faceAnalysis,cropFrame)
                # pool.spawn(visionAnalysis,frame)
                # pool.spawn(emotionAnalysis,frame)
        elif (status==SAVE_FRAME):
            print("say result")
            # cv2.imwrite("output/save.png", frame)
            idx = int(gResult["age"]) / 5
            if idx > len(siri.MALE_RESULT)-1:
                idx = len(siri.MALE_RESULT)-1
            if (gResult["gender"]=="male"):
                msg = siri.MALE_RESULT[idx]
                buf = msg.replace("{}", str(gResult["age"]))
            else:
                msg = siri.FEMALE_RESULT[idx]
                buf = msg.replace("{}", str(gResult["age"]))
            pool.spawn(say_result,buf)
            pool.spawn(show_image_text, "場次: {}\n姓名: {}\n帳號: {}\n預測年齡: {}".format(gResult['time'],gResult['username'],gResult['email'], str(gResult['age'])))
            logging.info("場次: {} 姓名: {} 帳號: {} 預測年齡: {}".format(gResult['time'],gResult['username'],gResult['email'], str(gResult['age'])))
            startTime = time.time()
        elif (status==SHOW_RESULT):
            # 顯示結果(3秒)
            if time.time()-startTime < 4:
                showCenterBottomText("{}:{},{}".format(gResult["username"],gResult["emotion"],str(int(time.time()-startTime))))
            if time.time()-startTime > 3:
                pool.spawn(say_bye, siri.SIRI_BYE[0])
        elif (status==END):
            showCenterBottomText("Thanks...")
            pass

        if ENABLE_FPS:
            print("fps:",t.fps())

        if status==DETECT_FACE:
            cv2.imshow('Video', frame_with_circle)
        else:
            cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cnt = cnt + 1
        gevent.sleep(0.001)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename="isobar.log", filemode="a+",format="%(asctime)-15s %(levelname)-8s %(message)s")
    print("click 'q' to quit")
    main()
