import tkinter as tk
import cv2
import threading
import time
import numpy as np
import requests
from datetime import datetime
from ultralytics import YOLO
import csv
import os
import config
import sys
from tkinter import font

lock = threading.Lock()

# Load YOLO model
model = YOLO('models/Model/weights/best.pt')

# Global flag to control threads
running = True

bad_conditions = ["mist", "thunderstorm", "rain", "shower rain"]

global info, flag_mini_frame, status, frames, start_time, messages, risk_zones, threads

global multiboat

multiboat = False

start_time = time.strftime("%H_%M_%S", time.gmtime(time.time()))
flag_mini_frame = False
threads = []

no_det = cv2.imread("utils/nodetect.png")

info = {
    "Camera Stream 1": [None, None],
    "Camera Stream 2": [None, None],
    "Camera Stream 3": [None, None],
    "Camera Stream 4": [None, None]
}

status = {
    "Camera Stream 1": [[], [], [], None],
    "Camera Stream 2": [[], [], [], None],
    "Camera Stream 3": [[], [], [], None],
    "Camera Stream 4": [[], [], [], None],
    "Total" : None
}


risk_zones = {
    "Camera Stream 1": [0,0,0,0,0],
    "Camera Stream 2": [0,0,0,0,0],
    "Camera Stream 3": [0,0,0,0,0],
    "Camera Stream 4": [0,0,0,0,0],
}


frames = {
    "Camera Stream 1": [None, False],
    "Camera Stream 2": [None, False],
    "Camera Stream 3": [None, False],
    "Camera Stream 4": [None, False]
}

logs = {
    "Camera Stream 1": [None, None, False],
    "Camera Stream 2": [None, None, False],
    "Camera Stream 3": [None, None, False],
    "Camera Stream 4": [None, None, False]
}

result = {
    "Camera Stream 1": [None, None],
    "Camera Stream 2": [None, None],
    "Camera Stream 3": [None, None],
    "Camera Stream 4": [None, None]
}


def zone_index(y_value):
    if 0 <= y_value <= 1:
        index = int(y_value / 0.20)
        return min(index, 4)  
    else:
        raise ValueError("Number is not in the range [0, 1]")

def zone():
    count_VIS = [0, 0, 0, 0, 0] 
    count_IR = [0, 0, 0, 0, 0]  

    for index in range(5):  
        for camera_stream in risk_zones:
            value = risk_zones[camera_stream][index]
            if value != 0:
                if "Camera Stream 1" in camera_stream or "Camera Stream 2" in camera_stream:
                    count_VIS[index] += risk_zones[camera_stream][index]
                elif "Camera Stream 3" in camera_stream or "Camera Stream 4" in camera_stream:
                    count_IR[index] += risk_zones[camera_stream][index]

    total = [max(count_VIS, count_IR) for count_VIS, count_IR in zip(count_VIS, count_IR)]
    return total


def risk_factor(frame,rframe):
    while running:
        for camera_stream in risk_zones:
            risk_zones[camera_stream] = [0, 0, 0, 0, 0]
        bad_conditions = ["overcast clouds", "mist", "shower rain", "rain", "thunderstorm"]
        risk = 0
        risk_value = 0
        cond = get_weather(False)
        if cond in bad_conditions:
            risk = bad_conditions.index(cond) + 1
        for camera_stream, value in status.items():
            if value is not None and len(value) > 2:
                centery = value[2]
                if isinstance(centery, list):
                    for item in centery:
                        index = zone_index(item)
                        risk_zones[camera_stream][index] += 1
        zones = zone()
        num_boat = 0
        for _, max_count in enumerate(zones):
            if max_count >= 2:
                num_boat = max_count
        risk_value = num_boat + risk
        risk_value = np.interp(risk_value, (0, 10), (0, 100))
        if risk_value < 33.0:
            color = "green1"
        elif 33.0 <= risk_value <= 66.0:
            color = "yellow1"
        else:
            color = "red1"
        frame.config(text = risk_value, bg=color, fg= "black", font=("Arial", 22))

def update_time(title):
    current_time = datetime.now().strftime("%A, %d/%m/%Y, %H:%M")
    name = current_time + ", Augusta:"
    title.config(text=name)
    root.after(60000,update_time,title)

def update_weather(l1,l2,r1,r2):
    while running:
        message = get_weather(True)
        mex = message.split("\n")        
        l1.config(text=mex[0])
        l2.config(text=mex[1])
        r1.config(text=mex[2])
        r2.config(text=mex[3])
        time.sleep(1)

def get_weather(call):
    base_url = "https://api.openweathermap.org/data/2.5/weather?"
    final_url = base_url + "appid=" + config.api_key + "&id=" + config.city_id + "&units=" + config.units
    try:
        response = requests.get(final_url)
        data = response.json()
        
        if response.status_code == 200:
            weather_description = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            
            message = (
                f"Weather: {weather_description}\n"
                f"Temperature: {temperature}°C\n"
                f"Humidity: {humidity}%\n"
                f"Wind Speed: {wind_speed} m/s\n"
            )
        else:
            print(f"Error: {data['message']}")
        if call:
            return message
        else:
            return weather_description
    except Exception as e:
        print(f"An error occurred: {e}")

def add_event(event_log, messages):
    text = "" 
    for key, value in messages.items():
        for i in range (int(len(value)/2)):

            if value[0] is None:
                text = text + key + " : No detection " + "\n"
            else:
                text = text + key + " : " + value[0] + "\n"
    event_log.config(text = text)

def add_frame(VFrame, IRFrame):
    flag = False
    if status["Total"] == "leaving":
        flag = True
        if frames["Camera Stream 1"][1] is not None:
            Vbbox = frames["Camera Stream 1"][1]
            Vfr = frames["Camera Stream 1"][0]
        else:
            Vbbox = frames["Camera Stream 2"][1]
            Vfr = frames["Camera Stream 2"][0]

        if frames["Camera Stream 3"][1] is not None:
            IRbbox = frames["Camera Stream 3"][1]
            IRfr = frames["Camera Stream 3"][0]
        else:
            IRbbox = frames["Camera Stream 4"][1]
            IRfr = frames["Camera Stream 4"][0]
    elif status["Total"] == "approaching":
        flag = True
        if frames["Camera Stream 2"][1] is not None:
            Vbbox = frames["Camera Stream 2"][1]
            Vfr = frames["Camera Stream 2"][0]
        else:
            Vbbox = frames["Camera Stream 1"][1]
            Vfr = frames["Camera Stream 1"][0]
        if frames["Camera Stream 4"][1] is not None:
            IRbbox = frames["Camera Stream 4"][1]
            IRfr = frames["Camera Stream 4"][0]
        else:
            IRbbox = frames["Camera Stream 3"][1]
            IRfr = frames["Camera Stream 3"][0]
    elif status["Total"] == None:
        flag = True
        IRfr = no_det
        Vfr = no_det
        Vbbox = None
        IRbbox = None

    if flag is True:
        if Vfr is not None:
            frame = Vfr
            image = tk.PhotoImage(data=cv2.imencode(".ppm", frame)[1].tobytes())
            VFrame.configure(image=image)
            VFrame.image = image
        if IRfr is not None:
            frame = IRfr
            image = tk.PhotoImage(data=cv2.imencode(".ppm", frame)[1].tobytes())
            IRFrame.configure(image=image)
            IRFrame.image = image
    return

            

def resize_frame(bbox, frame):
    if bbox is not None:
        bbox = bbox.astype(int)
        x1,y1,x2,y2 = np.reshape(bbox, (4,))
        frame = frame[y1:y2, x1:x2]
    target_width, target_height = 200, 80 
    # aspect_ratio = frame.shape[1] / frame.shape[0]
    # if aspect_ratio > target_width / target_height:
    #     target_height = int(target_width / aspect_ratio)
    # else:
    #     target_width = int(target_height * aspect_ratio)
    resized_frame = cv2.resize(frame, (target_width, target_height))
    return resized_frame



def calculate_box_center(corners):
    bbox = np.reshape(corners, (4,))
    xmin, ymin, xmax, ymax = bbox
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    return center_x, center_y

def determine_movement_direction(newx, newy, oldx, oldy2):
    delta_x = newx - oldx
    delta_y = newy - oldy2
    if delta_x < 0:
        if delta_y > 0:
            return "approaching" 
        else:
            return "leaving"
    else:
        if delta_y < 0:
            return "leaving"
        else:
            return "approaching"


def save_log(data):
    filename = start_time + "_log.csv" 
    file_exists = os.path.exists(filename)
    fieldnames = "Camera,Label,Time,Direction of travel"
    with open(filename, "a") as file:
        if not file_exists:
            file.write(fieldnames)
            file.write("\n")
        file.write(data)
        file.write("\n")


def detect_objects(frame):

    bbox = []
    labels = []
    results = model(frame, device=0, imgsz=(320,352), verbose=False)

    if len(results[0].boxes) != 0:
        for i in range (len(results[0].boxes)):
            labels.append(results[0].names[int(results[0].boxes.cls[i])])
            bbox.append(results[0].boxes.xyxy[i].cpu().numpy())
        return results[0], True, bbox, labels

    return frame, False, None, None

def display_camera_stream(camera_address, quadrant, event_log, camera_name):
    
    while running:
        cap = cv2.VideoCapture(camera_address)
        if not cap.isOpened():
            print(f"Failed to open camera: {camera_address}. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        while running:
            ret, frame = cap.read()
            
            if ret:
                with lock:  
                    frame, flag, result[camera_name][0], result[camera_name][1] = detect_objects(frame)  # Perform object detection on the frame

                if flag:
                    info[camera_name][1] = ""
                    if (len(result[camera_name][1]) > 1):
                        multiboat = True
                        for i in range (len(result[camera_name][0])):
                            box = result[camera_name][0][i]
                            label =  result[camera_name][1][i]
                            box_norm = frame.boxes.xyxyn[i].cpu().numpy()
                            _, zone_y = calculate_box_center(box_norm)
                            centerx, centery = calculate_box_center(box)
                            if len(status[camera_name][0]) < (len(result[camera_name][1])):
                                #! Devo salvare una lista per ogni x e y, utilizzare gli stessi e resettare quando finisco 
                                status[camera_name][0].append(centerx)
                                status[camera_name][1].append(centery)
                                status[camera_name][2].append(zone_y)
                                status[camera_name][3] = "detected"
                                status["Total"] = "detected"
                                # 
                                # logs[camera_name][0] = time.strftime("%H:%M:%S", time.gmtime(time.time()))
                                logs[camera_name][1] = label
                                if not status[camera_name][2]:
                                    status[camera_name][2].append(zone_y)
                                else:
                                    status[camera_name][2][i] = zone_y
                            else:
                                if (i == 0):
                                    frames[camera_name][0] = resize_frame(box, frame[0].orig_img)
                                    #frames[camera_name][0] = frame[0].orig_img
                                # else:
                                #     if frames[camera_name][0] is not None:
                                #         old_img = frames[camera_name][0]
                                #         new_img = resize_frame(box, frame[0].orig_img)
                                #         max_height = max(old_img.shape[0], new_img.shape[0])
                                #         max_width = max(old_img.shape[1], new_img.shape[1])
                                #         # Calculate padding for img1
                                #         top_pad_img1 = (max_height - old_img.shape[0]) // 2
                                #         left_pad_img1 = (max_width - old_img.shape[1]) // 2

                                #         # Calculate padding for img2
                                #         top_pad_img2 = (max_height - new_img.shape[0]) // 2
                                #         left_pad_img2 = (max_width - new_img.shape[1]) // 2

                                #         # Create padded images
                                #         img1 = cv2.copyMakeBorder(src=old_img,
                                #                                 top=0,
                                #                                 bottom=max_height-old_img.shape[0],
                                #                                 left=0,
                                #                                 right=max_width-old_img.shape[1],
                                #                                 borderType=cv2.BORDER_CONSTANT,
                                #                                 value=[255, 255, 255])

                                #         img2 = cv2.copyMakeBorder(src=new_img,
                                #                                 top=0,
                                #                                 bottom=max_height-new_img.shape[0],
                                #                                 left=0,
                                #                                 right=max_width-new_img.shape[1],
                                #                                 borderType=cv2.BORDER_CONSTANT,
                                #                                 value=[255, 255, 255])
                                #         img_add_v = cv2.vconcat([img1,img2]) 
                                #         frames[camera_name][0] = img_add_v
                                frames[camera_name][1] = box
                                oldx = status[camera_name][0][i]
                                oldy = status[camera_name][1][i]
                                status[camera_name][0][i] = centerx
                                status[camera_name][1][i] = centery
                                status[camera_name][2][i] = zone_y
                                #print(camera_name, "n° ",i, "xdiff: ", oldx - centerx,"ydiff:", oldy - centery)
                                status[camera_name][3] = determine_movement_direction(centerx, centery, oldx, oldy)
                                status["Total"] = status[camera_name][3]
                                if logs[camera_name][2] is False:
                                    logs[camera_name][2] = True
                                    # data = camera_name + "," + logs[camera_name][1] + "," + logs[camera_name][0] + "," + status[camera_name][3]
                                    # save_log(data)
                            message = "A " + label + " is " + status[camera_name][3]
                            if not info[camera_name][1]:
                                info[camera_name][1] = message
                            else:
                                info[camera_name][1] = info[camera_name][1] + " & " + message
                            info[camera_name][0] = info[camera_name][1]
                        frame = frame.plot()
                    else:
                        # multiboat = False
                        box = frame[0].boxes.xyxy[0].cpu().numpy()
                        label = frame[0].names[int(frame[0].boxes.cls[0])]
                        centerx, centery = calculate_box_center(box)
                        box_norm = frame.boxes.xyxyn[0].cpu().numpy()
                        _, zone_y = calculate_box_center(box_norm)
                        if not status[camera_name][0]:
                            status[camera_name][0].append(centerx)
                            status[camera_name][1].append(centery)
                            status[camera_name][3] = "detected"
                            status["Total"] = "detected"
                            #logs[camera_name][0] = time.strftime("%H:%M:%S", time.gmtime(time.time()))
                            logs[camera_name][1] = label
                            if not status[camera_name][2]:
                                status[camera_name][2].append(zone_y)
                            else:
                                status[camera_name][2][0] = zone_y
                        else:
                            frames[camera_name][0] = resize_frame(box, frame[0].orig_img)
                            frames[camera_name][1] = box
                            oldx = status[camera_name][0][0]
                            oldy = status[camera_name][1][0]
                            status[camera_name][0][0] = centerx
                            status[camera_name][1][0] = centery
                            status[camera_name][2][0] = zone_y
                            #print(camera_name, "xdiff: ", oldx - centerx,"ydiff:", oldy - centery)
                            status[camera_name][3] = determine_movement_direction(centerx, centery, oldx, oldy)
                            status["Total"] = status[camera_name][3]
                            if logs[camera_name][2] is False:
                                logs[camera_name][2] = True
                                # data = camera_name + "," + logs[camera_name][1] + "," + logs[camera_name][0] + "," + status[camera_name][3]
                                # save_log(data)
                        message = "A " + label + " is " + status[camera_name][3]
                        info[camera_name][0] =  message
                        frame = frame.plot()
                    threading.Thread(target=add_frame, args=(VFrame,IRFrame), daemon=True).start()
                else:
                    frames[camera_name][0] = None
                    frames[camera_name][1] = None
                    info[camera_name] = [None, None]
                    for i in range(3):
                        status[camera_name][i] = []
                # ! Reset quando non vedo nulla
                if (all(value is None for value in info.values())):
                    status["Total"] = None
                    # for camera_stream in status:
                    #     for i in range(3):
                    #         status[camera_stream][i] = []
                    for key in logs:
                        logs[key][2] = False

                target_width, target_height = 320, 352  # Adjust as needed
                aspect_ratio = frame.shape[1] / frame.shape[0]
                if aspect_ratio > target_width / target_height:
                    target_height = int(target_width / aspect_ratio)
                else:
                    target_width = int(target_height * aspect_ratio)
                resized_frame = cv2.resize(frame, (target_width, target_height))
                image = tk.PhotoImage(data=cv2.imencode(".ppm", resized_frame)[1].tobytes()) #Exception has occurred: RuntimeError Too early to create image: no default root window
                quadrant.configure(image=image)
                quadrant.image = image
                add_event(event_log, info)
            else:
                print(f"Failed to read frame from camera: {camera_address}. Retrying...")
                cap.release()
                break

        cap.release()



def create_gui(root):

    # # Main area divided into 4 quadrants
    q1_frame = tk.LabelFrame(root, text="Camera Stream 1:", width=3,height=3, labelanchor="n", font=font.Font(weight="bold"))
    q1_frame.pack_propagate(0)
    c1 = tk.Frame(q1_frame)
    c1.pack(side="top", padx=10, pady=10)
    quadrant_1 = tk.Label(c1, bg="grey")
    quadrant_1.pack(anchor="center")
    
    q2_frame = tk.LabelFrame(root, text="Camera Stream 2:", width=3,height=3, labelanchor="n", font=font.Font(weight="bold"))
    q2_frame.pack_propagate(0)
    c2 = tk.Frame(q2_frame)
    c2.pack(side="top", padx=10, pady=10)
    quadrant_2 = tk.Label(c2, bg="grey")
    quadrant_2.pack(anchor="center")

    q3_frame = tk.LabelFrame(root, text="Camera Stream 3:", width=3,height=3, labelanchor="n", font=font.Font(weight="bold"))
    q3_frame.pack_propagate(0)
    c3 = tk.Frame(q3_frame)
    c3.pack(side="top", padx=10, pady=10)
    quadrant_3 = tk.Label(c3, bg="grey")
    quadrant_3.pack(anchor="center")


    q4_frame = tk.LabelFrame(root, text="Camera Stream 4:", width=3,height=3, labelanchor="n", font=font.Font(weight="bold"))
    q4_frame.pack_propagate(0)
    c4 = tk.Frame(q4_frame)
    c4.pack(side="top", padx=10, pady=10)
    quadrant_4 = tk.Label(c4, bg="grey")
    quadrant_4.pack(anchor="center")

    # Event log frame
    event_log_frame = tk.LabelFrame(root, text="Event Log:", width=400, height=200,  labelanchor="n", font=font.Font(weight="bold"))
    event_log_frame.pack_propagate(0)
    columns = tk.Frame(event_log_frame)
    columns.pack(side="top", padx=5, pady=5)
    elog = tk.Label(columns, wraplength=350)
    elog.pack(anchor="ne")

    # Additional information in two columns
    additional_info_frame = tk.LabelFrame(root, text="Weather in Augusta:", width=60, labelanchor="n", font=font.Font(weight="bold"))
    left_column = tk.Frame(additional_info_frame)
    left_column.pack(side="left", padx=5, pady=5)
    right_column = tk.Frame(additional_info_frame)
    right_column.pack(side="right", padx=5, pady=5)
    left_label1 = tk.Label(left_column)
    left_label1.pack(anchor="w")
    left_label2 = tk.Label(left_column)
    left_label2.pack(anchor="w")
    right_label1 = tk.Label(right_column)
    right_label1.pack(anchor="w")
    right_label2 = tk.Label(right_column)
    right_label2.pack(anchor="w")

    # Visibile and IR Cam frames
    global VFrame, IRFrame
    visible_frame = tk.LabelFrame(root, text="Visibile Cam:", width=225, height=225, labelanchor="n", font=font.Font(weight="bold"))
    visible_frame.pack_propagate(0)
    frame = tk.Frame(visible_frame)
    frame.pack(side="top", padx=5, pady=5)
    VFrame = tk.Label(frame)
    VFrame.pack(anchor="center")

    IR_frame = tk.LabelFrame(root, text="IR Cam:", width=225, height=225, labelanchor="n", font=font.Font(weight="bold"))
    IR_frame.pack_propagate(0)
    frame2 = tk.Frame(IR_frame)
    frame2.pack(side="top", padx=5, pady=5)
    IRFrame = tk.Label(frame2)
    IRFrame.pack(anchor="center")


    #Risk indicator
    Risk_frame = tk.LabelFrame(root, text="Risk indicator:", width=144, height=72, labelanchor="n", font=font.Font(weight="bold"))
    Risk_frame.pack_propagate(0)
    frame3 = tk.Frame(Risk_frame)
    frame3.pack(side="top", padx=2, pady=2)
    RiskFrame = tk.Label(frame3, anchor="center")
    RiskFrame.pack()
    
    Risk_reason_frame = tk.LabelFrame(root, text="Reason:", width=144, height=72, labelanchor="n", font=font.Font(weight="bold"))
    Risk_reason_frame.pack_propagate(0)
    frame4 = tk.Frame(Risk_reason_frame)
    frame4.pack(side="top", padx=2, pady=2)
    RiskReason = tk.Label(frame4, anchor="center")
    RiskReason.pack()


    root.grid_rowconfigure(0, weight=4)
    root.grid_rowconfigure(1, weight=4)
    root.grid_columnconfigure(0, weight=5)
    root.grid_columnconfigure(1, weight=3)
    root.grid_columnconfigure(2, weight=1)
    root.grid_rowconfigure(2, weight=1)

    q1_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    q2_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
    q3_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
    q4_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
    event_log_frame.grid(row=0, column=2, rowspan=2, columnspan=1, padx=0, pady=5, sticky="n")
    additional_info_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
    visible_frame.grid(row=1, column=2,rowspan=2, columnspan=1, padx=0, pady=5, sticky="nw")
    IR_frame.grid(row=1, column=2, rowspan=2 ,padx=0, pady=5, sticky="ne")

    Risk_frame.grid(row=2, column=1, columnspan=1, padx=5, pady=5, sticky="w")
    Risk_reason_frame.grid(row=2, column=1, columnspan=1, padx=5, pady=5, sticky="e")

    
    multicameras = {
        "Camera Stream 1": "Video/Multi/cam1.mp4",
        "Camera Stream 2": "Video/Multi/cam2.mp4",
        "Camera Stream 3": "Video/Multi/cam3.mp4",
        "Camera Stream 4": "Video/Multi/cam4.mp4"
    }
    cameras = {
        "Camera Stream 1": "rtsp://192.168.71.11:554/stream1",
        "Camera Stream 2": "rtsp://192.168.71.12:554/stream1",
        "Camera Stream 3": "rtsp://admin:Admin2022%23@192.168.71.14:554/ch0",
        "Camera Stream 4": "rtsp://admin:Admin2022%23@192.168.71.13:554/ch0"
    }

    mareforte = {
        "Camera Stream 1": "Video/05_09_2023_09_34_29/Cam1.mkv",
        "Camera Stream 2": "Video/05_09_2023_09_34_29/Cam2.mkv",
        "Camera Stream 3": "Video/05_09_2023_09_34_29/Cam3.mkv",
        "Camera Stream 4": "Video/05_09_2023_09_34_29/Cam4.mkv"
    }

    notturno = {
        "Camera Stream 1": "Video/15_09_2023 00_50_43/ir/Cam1.mp4",
        "Camera Stream 2": "Video/15_09_2023 00_50_43/ir/Cam2.mp4",
        "Camera Stream 3": "Video/15_09_2023 00_50_43/old/Cam3.mkv",
        "Camera Stream 4": "Video/15_09_2023 00_50_43/old/Cam4.mkv"
    }

    for camera_name, camera_address in cameras.items():
        quadrant = locals()[f"quadrant_{camera_name.split()[-1]}"]
        thread = threading.Thread(target=display_camera_stream, args=(camera_address, quadrant, elog, camera_name), daemon=True)
        threads.append(thread)
    threads.append(threading.Thread(target=update_weather, args=(left_label1, left_label2, right_label1, right_label2), daemon=True))
    threads.append(threading.Thread(target=update_time, args=(additional_info_frame,), daemon=True))
    threads.append(threading.Thread(target=risk_factor, args=(RiskFrame,RiskReason), daemon=True))
    for thread in threads:
        thread.start()


def on_closing():
    global running
    running = False  # Set the global flag to stop threads
    time.sleep(2)
    root.destroy()   # Destroy the main GUI window
    sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Camera Streams GUI")
    root.geometry("1200x800")
    icon = tk.PhotoImage(file="utils/icon.png")
    root.iconphoto(True, icon) 
    create_gui(root)

    root.protocol("WM_DELETE_WINDOW", on_closing)  # Intercept window close event

    root.mainloop()

    # After the main loop, wait for threads to finish
    for thread in threading.enumerate():
        if thread != threading.current_thread():
            thread.join()
    