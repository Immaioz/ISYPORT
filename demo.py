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

lock = threading.Lock()

# Load YOLO model
model = YOLO('models/Model/weights/best.pt')

# Global flag to control threads
running = True

global info, old_centerx, status, frames, start_time, messages
start_time = time.strftime("%H_%M_%S", time.gmtime(time.time()))

no_det = cv2.imread("utils/nodetect.png")

info = {
    "Camera Stream 1": [None, None],
    "Camera Stream 2": [None, None],
    "Camera Stream 3": [None, None],
    "Camera Stream 4": [None, None]
}

status = {
    "Camera Stream 1": [None, None, None],
    "Camera Stream 2": [None, None, None],
    "Camera Stream 3": [None, None, None],
    "Camera Stream 4": [None, None, None],
    "Total" : None
}

frames = {
    "Camera Stream 1": [None, None],
    "Camera Stream 2": [None, None],
    "Camera Stream 3": [None, None],
    "Camera Stream 4": [None, None]
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


def update_time(title):
    current_time = datetime.now().strftime("%A, %d/%m/%Y, %H:%M")
    name = current_time + ", Augusta:"
    title.config(text=name)
    root.after(60000,update_time,title)

def update_weather(l1,l2,r1,r2):
    while running:
        message = get_weather()
        mex = message.split("\n")        
        l1.config(text=mex[0])
        l2.config(text=mex[1])
        r1.config(text=mex[2])
        r2.config(text=mex[3])
        time.sleep(10)

def get_weather():
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
        return message
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
    while running:
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
                frame = resize_frame(Vbbox, Vfr)
                image = tk.PhotoImage(data=cv2.imencode(".ppm", frame)[1].tobytes())
                VFrame.configure(image=image)
                VFrame.image = image
            if IRfr is not None:
                frame = resize_frame(IRbbox, IRfr)
                image = tk.PhotoImage(data=cv2.imencode(".ppm", frame)[1].tobytes())
                IRFrame.configure(image=image)
                IRFrame.image = image

def resize_frame(bbox, frame):
    if bbox is not None:
        bbox = bbox.astype(int)
        x1,y1,x2,y2 = np.reshape(bbox, (4,))
        frame = frame[y1:y2, x1:x2]
    target_width, target_height = 151, 72  # Adjust as needed
    aspect_ratio = frame.shape[1] / frame.shape[0]
    if aspect_ratio > target_width / target_height:
        target_height = int(target_width / aspect_ratio)
    else:
        target_width = int(target_height * aspect_ratio)
    resized_frame = cv2.resize(frame, (target_width, target_height))
    return resized_frame



def calculate_box_center(corners):
    bbox = np.reshape(corners, (4,))
    xmin, ymin, xmax, ymax = bbox
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    return center_x, center_y

def determine_movement_direction(x1, y1, x2):
    delta_x = x2 - x1
    if delta_x > 0:
        return "approaching"
    elif delta_x < 0:
        return "leaving"

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
    results = model(frame, device=0, imgsz=(320,352), verbose=False, iou=0.9, conf=0.5)

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
                        for i in range (len(result[camera_name][0])):
                            box = result[camera_name][0][i]
                            label =  result[camera_name][1][i]
                            centerx, centery = calculate_box_center(box)
                            if status[camera_name][0] == None:
                                status[camera_name][0] = centerx
                                status[camera_name][1] = centery
                                status[camera_name][2] = "detected"
                                status["Total"] = "detected"
                                # logs[camera_name][0] = time.strftime("%H:%M:%S", time.gmtime(time.time()))
                                logs[camera_name][1] = label
                            else:
                                if (i == 0):
                                    frames[camera_name][0] = frame[0].orig_img.copy()
                                frames[camera_name][1] = box
                                old = status[camera_name][0]
                                status[camera_name][2] = determine_movement_direction(centerx, centery, old)
                                status["Total"] = status[camera_name][2]
                                if logs[camera_name][2] is False:
                                    logs[camera_name][2] = True
                                    # data = camera_name + "," + logs[camera_name][1] + "," + logs[camera_name][0] + "," + status[camera_name][2]
                                    # save_log(data)
                            message = "A " + label + " is " + status[camera_name][2]
                            if not info[camera_name][1]:
                                info[camera_name][1] = message
                            else:
                                info[camera_name][1] = info[camera_name][1] + " & " + message
                            info[camera_name][0] = info[camera_name][1]
                        frame = frame.plot()
                    else:
                        box = frame[0].boxes.xyxy[0].cpu().numpy()
                        label = frame[0].names[int(frame[0].boxes.cls[0])]
                        centerx, centery = calculate_box_center(box)
                        
                        if status[camera_name][0] == None:
                            status[camera_name][0] = centerx
                            status[camera_name][1] = centery
                            status[camera_name][2] = "detected"
                            status["Total"] = "detected"
                            #logs[camera_name][0] = time.strftime("%H:%M:%S", time.gmtime(time.time()))
                            logs[camera_name][1] = label
                        else:
                            frames[camera_name][0] = frame[0].orig_img.copy()
                            frames[camera_name][1] = box
                            old = status[camera_name][0]
                            status[camera_name][2] = determine_movement_direction(centerx, centery, old)
                            status["Total"] = status[camera_name][2]
                            if logs[camera_name][2] is False:
                                logs[camera_name][2] = True
                                # data = camera_name + "," + logs[camera_name][1] + "," + logs[camera_name][0] + "," + status[camera_name][2]
                                # save_log(data)
                        message = "A " + label + " is " + status[camera_name][2]
                        info[camera_name][0] =  message
                        frame = frame.plot()
                else:
                    frames[camera_name] = [None, None]
                    info[camera_name] = [None, None]

                if (all(value is None for value in info.values())):
                    status["Total"] = None
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
    # Main area divided into 4 quadrants
    quadrant_1 = tk.Label(root, bg="grey", width=60, height=30)  # Larger size for higher resolution
    quadrant_1.pack_propagate(0)
    quadrant_2 = tk.Label(root, bg="grey", width=60, height=30)  # Larger size for higher resolution
    quadrant_2.pack_propagate(0)
    quadrant_3 = tk.Label(root, bg="grey", width=60, height=30)  # Larger size for higher resolution
    quadrant_3.pack_propagate(0)
    quadrant_4 = tk.Label(root, bg="grey", width=60, height=30)  # Larger size for higher resolution
    quadrant_4.pack_propagate(0)

    # Event log frame
    event_log_frame = tk.LabelFrame(root, text="Event Log:", width=400, height=300,  labelanchor="n")
    event_log_frame.pack_propagate(0)
    columns = tk.Frame(event_log_frame)
    columns.pack(side="top", padx=5, pady=5)
    elog = tk.Label(columns, wraplength=350)
    elog.pack(anchor="ne")

    # Additional information in two columns
    additional_info_frame = tk.LabelFrame(root, text="Weather in Augusta:", width=60, labelanchor="n")
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

    root.grid_rowconfigure(0, weight=3)
    root.grid_rowconfigure(1, weight=3)
    root.grid_columnconfigure(0, weight=3)
    root.grid_columnconfigure(1, weight=3)
    root.grid_columnconfigure(2, weight=1)
    root.grid_rowconfigure(2, weight=1)

    # Visibile and IR Cam frames
    visible_frame = tk.LabelFrame(root, text="Visibile Cam:", width=155, height=80, labelanchor="n")
    visible_frame.pack_propagate(0)
    frame = tk.Frame(visible_frame)
    frame.pack(side="top", padx=2, pady=2)
    VFrame = tk.Label(frame)
    VFrame.pack(anchor="e")

    IR_frame = tk.LabelFrame(root, text="IR Cam:", width=155, height=80, labelanchor="n")
    IR_frame.pack_propagate(0)
    frame2 = tk.Frame(IR_frame)
    frame2.pack(side="top", padx=2, pady=2)
    IRFrame = tk.Label(frame2)
    IRFrame.pack(anchor="e")

    #Risk indicator
    Risk_frame = tk.LabelFrame(root, text="Risk indicator:", width=151, height=72, labelanchor="n")
    Risk_frame.pack_propagate(0)
    frame3 = tk.Frame(Risk_frame)
    frame3.pack(side="top", padx=2, pady=2)
    RiskFrame = tk.Label(frame3)
    RiskFrame.pack(anchor="s")

    quadrant_1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    quadrant_2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
    quadrant_3.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
    quadrant_4.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
    event_log_frame.grid(row=0, column=2, rowspan=2, columnspan=4, padx=0, pady=5, sticky="n")
    # event_log_frame_2.grid(row=0, column=5, rowspan=2, columnspan=4, padx=0, pady=5, sticky="n")
    additional_info_frame.grid(row=2, column=0, columnspan=4, padx=20, pady=0, sticky="w")
    visible_frame.grid(row=1, column=3, padx=50, pady=0, sticky="")
    IR_frame.grid(row=1, column=2, padx=0, pady=0, sticky="")
    Risk_frame.grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky="")
    # visible_frame.grid(row=2, column=0,columnspan=1, padx=5, pady=5, sticky="")
    # IR_frame.grid(row=2, column=1, columnspan=1, padx=5, pady=5, sticky="")
    
    cameras = {
        "Camera Stream 1": "Video/test2/Cam1.mp4",
        "Camera Stream 2": "Video/test2/Cam2.mp4",
        "Camera Stream 3": "Video/test2/Cam3.mp4",
        "Camera Stream 4": "Video/test2/Cam4.mp4"
    }

    for camera_name, camera_address in cameras.items():
        quadrant = locals()[f"quadrant_{camera_name.split()[-1]}"]
        threading.Thread(target=display_camera_stream, args=(camera_address, quadrant, elog, camera_name), daemon=True).start()
    threading.Thread(target=add_frame, args=(VFrame,IRFrame), daemon=True).start()
    threading.Thread(target=update_weather, args=(left_label1, left_label2, right_label1, right_label2), daemon=True).start()
    threading.Thread(target=update_time, args=(additional_info_frame,), daemon=True).start()

def on_closing():
    global running
    running = False  # Set the global flag to stop threads
    root.destroy()   # Destroy the main GUI window

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