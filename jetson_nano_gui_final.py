#connect_to_mysqql_movie_booking
import pymysql
import sys, os, dlib, glob
import numpy as np
from skimage import io
import imutils
import cv2
import json
import os
from urllib.request import urlretrieve
from PIL import Image,ImageFont,ImageDraw
# movie='永恆族'
# session='第一場'




config = {
        'host': '35.229.212.128',
        'port': 3306,
        'user': 'ortonrocks',
        'passwd': 'jimmy19971027',
        'db': 'try',
        'charset': 'utf8mb4',
        'local_infile': 1

    }



def audience_determine(uname,mname,session):

    conn = pymysql.connect(**config)
    print('successfully connected')

    cur = conn.cursor()

    get_data_sql = "select uname from movie_booking where ( mname='{}') and ( session='{}') ;".format(mname,session)
    cur.execute(get_data_sql)
    audience=data=cur.fetchall()
    cur.close() 
    conn.close()
    
    for i in range(len(audience)):
        if audience[i][0]==uname:
            
            return "歡迎觀眾：{}入場".format(uname)
        
        else:
            pass
    return "您不是本場次觀眾！"




def download():
    directory = "C:\\Users\\Tibame_T14\\Documents"
    url = 'https://trymoviehelper.tk/downloads_json'
    #directory ="./jetson_nano/user_face_info.json"
    if os.path.isfile(directory):
        os.remove(directory)    
    urlretrieve(url,directory)



def detection(frame,movie_name,movie_session):
    print(movie_name)
    result_name=''

    #ret, frame = cap.read()
    dets = detector(frame, 1)

    distance = []

    for k, d in enumerate(dets):
        # 68特徵點偵測
        shape = shape_predictor(frame, d)

        # 128維特徵向量描述子

        face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)

        # 轉換 numpy array 格式
        d_test = np.array(face_descriptor)  # 攝像機所照到人臉的特徵向量(已轉成np array格式)
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        # 以方框框出人臉
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

        # 計算歐式距離，線性運算函式
        for i in descriptors:
            dist_ = np.linalg.norm(i - d_test)  # 把人臉資料庫的每張人臉的特徵向量跟攝像機所照到的人臉特徵向量 拿來算歐式距離
            distance.append(dist_)  # 把歐式距離存在distance這個list

        # 利用zip函數將人臉名稱跟歐式距離元素打包成一個tuple
        # 並存入dict(候選圖片,距離)
        candidate_distance_dict = dict(zip(candidate, distance))  # 資料格式為 {圖片名稱 : 資料庫人臉跟攝像機人臉的歐式距離}

        # 接著將候選圖片及人名進行排序
        candidate_distance_dict_sorted = sorted(candidate_distance_dict.items(),
                                                key=lambda d: d[1])  # [i][0]=人名 [i][1]=距離，第一個維度是代表第 i 張人臉圖片
        pic_fit = False  # 尚未有符合人臉圖片庫的人臉 pic_fit設為False，若有符合則為True
        for i in range(len(candidate_distance_dict_sorted)):
            pic_distance = candidate_distance_dict_sorted[i][1]
            if pic_distance < setting_distance:  # 當圖片的人臉特徵距離 < 所設定的筏值，代表找到符合人臉圖片庫內的人臉
                pic_fit = True
        if pic_fit == True:  # 找到符合的人臉
            try:
                result = candidate_distance_dict_sorted[0][0]  # 取出最短距離維辨識出的對象 #不知道為什麼會 index out of range
                face_name_split = result.split("_")  # 把人臉照片的編號拿掉
                face_name = face_name_split[0]  # 取出人名
                

            except IndexError:
                pass
            
            


            # 當偵測用戶時,輸出用戶名，克服中文編碼問題
            # cv2讀取影片
            cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(cv2img)

            # PIL圖片上輸出中文
            draw = ImageDraw.Draw(pilimg)
            font = ImageFont.truetype("SimHei.ttf", 30, encoding='utf-8')
            
            draw.text((0, 20),audience_determine(face_name,movie_name,movie_session) , (255, 0, 0), font=font)
            
            draw.text((x1, y1 - 25), face_name, (255, 0, 0), font=font)

            # PIL轉cv2
            frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
            #frame = imutils.resize(frame, width=500)



                # 在方框旁邊標上人名
            #cv2.putText(frame, face_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            #frame = imutils.resize(frame, width=500)
        else:  # pic_fit == False
            unknown_result = "Undefined User!"

            try:  # 排除抓不到人臉會出現的Error
                cv2.putText(frame, unknown_result, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)

            except NameError:
                pass
            
    
#     cap.release()
#     cv2.destroyAllWindows()

    return(frame,result_name)



# 人臉偵測部分 : 將人臉資料庫的特徵json導入 再逐一算出歐式距離 找出最近的歐式距離 #

#### 更新人臉特徵向量表 ####
# download()

# 讀取人臉名稱與特徵向量的檔案 { 人臉名稱 : 128維向量 }
#設定人臉特徵距離筏值 0.4:嚴格 0.5:寬鬆
setting_distance = 0.45
# 開啟影片檔案
# 載入人臉檢測器
detector = dlib.get_frontal_face_detector()
# 人臉68特徵點模型的路徑及檢測器
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 載入人臉辨識模型及檢測器
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
# 人臉描述子list
descriptors = []
# 候選人臉名稱list
candidate = []


with open("user_face_info.json", 'r', encoding="UTF-8") as load_f:
    name_vector_json = json.load(load_f)  # json檔案讀進來會是str型態

name_vector_dict = json.loads(name_vector_json)  # 將str型態轉成dict

candidate = list(name_vector_dict.keys())

face_descriptors = list(name_vector_dict.values())

# 將face_descriptors裡面的每一個128維向量 轉換成 numpy array 格式 再存進descriptors
for face_descriptor in face_descriptors:
    v = np.array(face_descriptor)
    descriptors.append(v)

# 對要辨識的目標圖片作處理
# 讀取照片








import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
#from tkinter import *
from tkinter import Entry
import sys, os, dlib, glob
import numpy as np
from skimage import io
import imutils
import cv2
import json
import os
from urllib.request import urlretrieve
from PIL import Image,ImageFont,ImageDraw
import time

#剪票頁面設計
def createNewWindow():
    def show_frame():
        time1=time.time()
        
        
        _, frame = cap.read()
        frame=detection(frame,movie_name,movie_session)[0]
        #_, frame = True,frame
        print(frame.shape)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(1, show_frame) 
        time2=time.time()
        print(time2-time1)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    cap.set(cv2.CAP_PROP_FPS, 25)       
    movie_name=movie_name_input.get()
    movie_session=movie_session_input.get()
#     print(movie_name)
#     print(movie_session)
    

    
    _,frame=cap.read()
    #print(detection(frame)[1])
    window = tk.Toplevel(app)
    window.geometry("700x1000")
    window.wm_title("剪票系統")
    window.config(background="#FFFFFF")


    label1=tk.Label(window,text='AI剪票助手',bg="#FFFFFF",font=("宋體", 32))
    label1.grid(row=0,column=0)
    label2=tk.Label(window,text='電影場次：{}{}'.format(movie_name,movie_session),bg="#FFFFFF",font=("宋體", 32))
    label2.grid(row=0,column=0)
    label2=tk.Label(window,text='人臉偵測中...',bg="#FFFFFF",font=("宋體", 24)).grid(row=1,column=0,sticky=tk.E+tk.W)

    #Graphics window
    imageFrame = tk.Frame(window, width=320, height=180,bg="#FFFFFF")
    imageFrame.grid(row=3, column=0,sticky=tk.E+tk.W)

    #Capture video frames
    lmain = tk.Label(imageFrame)
    lmain.grid(row=2, column=0,sticky=tk.E+tk.W)
    
    
    
    
    
    


    
    show_frame()  #Display 2
    window.mainloop()  #Starts GUI
    cap.release()
    cv2.destroyAllWindows()

    
    
#登錄頁面設計
app = tk.Tk()
app.geometry("400x300")
app.wm_title("AI剪票助手")
app.config(background="#FFFFFF")
title=tk.Label(app,text='AI剪票助手',bg="#FFFFFF",font=("宋體", 32))
title.grid(row=0,column=0,columnspan=2,sticky=tk.N+tk.S)



movie_name_label=tk.Label(app,text='輸入本場電影名稱：',bg="#FFFFFF",font=("宋體", 18))
movie_name_label.grid(row=1,column=0,sticky=tk.N+tk.S,pady=5)



movie_name_input= Entry(app, bd =2,font=("宋體", 12))
movie_name_input.grid(row=1,column=1,sticky=tk.N+tk.S)


movie_session_label=tk.Label(app,text='輸入電影場次：',bg="#FFFFFF",font=("宋體", 18))
movie_session_label.grid(row=2,column=0,sticky=tk.N+tk.S,pady=5)

movie_session_input= Entry(app, bd =2,font=("宋體", 12))
movie_session_input.grid(row=2,column=1,sticky=tk.N+tk.S)
movie_session=movie_session_input.get()

confirm_button = tk.Button(app, text="開始剪票",command=createNewWindow)
confirm_button.grid(row=3,column=0,columnspan=2,sticky=tk.N+tk.S)

app.mainloop()




