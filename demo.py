import tkinter as tk
import cv2
import threading
import time
import numpy as np
import requests
from datetime import datetime
from ultralytics import YOLO
import config
import sys
from tkinter import font
from tkinter import ttk
import math 

from Boat import Boat
from ToolTip import ToolTip


lock = threading.Lock()

modelVIS = YOLO('models/VISModel.pt')
modelIR = YOLO('models/IRModel.pt')


names = modelVIS.names
direction = ["leaving", "approaching"]

bad_conditions = ["mist", "thunderstorm", "rain", "shower rain"]

global threads, event_running, frame_running

running = True
event_running = False
frame_running = False

threads = []

no_det = cv2.imread("utils/nodetect.png")

detected = {
    "Camera Stream 1": [],
    "Camera Stream 2": [],
    "Camera Stream 3": [],
    "Camera Stream 4": []
}


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def search_nearest(target,c_name):
    min_distance = float('inf')
    max_distance = 200
    items = detected[c_name]
    found = False
    for i, item in enumerate(items):
        dist = distance(item.pos, target)
        if dist < min_distance and dist <= max_distance:
            min_distance = dist
            found = True
            found_pos = i
    if found:
        return found_pos, True
    else:
        return i, False

def update_scroll_region_vis():
    scrollbarVIS.config(scrollregion=scrollbarVIS.bbox("all"))

def update_scroll_region_ir():
    scrollbarIR.config(scrollregion=scrollbarIR.bbox("all"))

def rgb(r, g, b):
    return "#%s%s%s" % tuple([hex(c)[2:].rjust(2, "0") for c in (r, g, b)])

def draw_gradient(canvas):
    for x in range(0, 256):
        r = x * 2 if x < 128 else 255
        g = 255 if x < 128 else 255 - (x - 128) * 2
        canvas.create_rectangle(x*2, 5, x*2 + 2, 50, fill=rgb(r, g, 0), outline=rgb(r, g, 0))

def resize_canvas(canvas):
    for item in canvas.find_all():
        new_coords = [canvas.coords(item)[0]/3, canvas.coords(item)[1], canvas.coords(item)[2]/3, canvas.coords(item)[3]]
        canvas.coords(item, *new_coords)
        w = canvas.coords(item)[2]
        h = canvas.coords(item)[3]
    return w,h

def draw_indicator(canvas, width, ini, mod):
    if mod is True:
        canvas.delete("top")
        canvas.delete("mid")
        canvas.delete("bot")
    pos = int((ini * (width)) /100) 
    points = [pos-8,0, pos+8,0, pos,10]
    canvas.create_polygon(points,  tags="top")
    points2 = [pos-8,55, pos+8,55, pos,45]
    canvas.create_polygon(points2,   tags="bot")
    line = [pos,5,pos,50]
    canvas.create_line(line, width=3, tags="mid")

def zone():
    count_VIS = [0, 0, 0, 0, 0] 
    count_IR = [0, 0, 0, 0, 0]  
    for cname in detected.keys():
        if detected[cname]:
            for boat in detected[cname]:
                index = boat.zone
                if "Camera Stream 1" in cname or "Camera Stream 2" in cname:
                    count_VIS[index] += 1
                elif "Camera Stream 3" in cname or "Camera Stream 4" in cname:
                    count_IR[index] += 1        

    total = np.max([max(count_VIS, count_IR) for count_VIS, count_IR in zip(count_VIS, count_IR)])
    return total


def risk_factor(frame,rframe, gradient, w):
    while running:
        bad_conditions = ["overcast clouds", "mist", "shower rain", "rain", "thunderstorm"]
        risk = 0
        risk_value = 0
        cond, wind = get_weather(False)
        
        if cond in bad_conditions:
            risk = bad_conditions.index(cond) + 1
        
        if wind > 7.0:
            temp = wind - 7.0
            temp = (temp//3.0) + 1
        else:
            temp = 0

        risk = risk + temp # max 13

        num_boat = zone()  # max ipotetico 3/4 barche 
        risk_value = num_boat + risk

        risk_value = int(np.interp(risk_value, (0, 17), (0, 100)))
        if risk_value < 33.0:
            color = "green1"
            reason = "Optimal Conditions"
        elif 33.0 <= risk_value <= 66.0:
            color = "yellow1"
            if risk > num_boat:
                reason = "Adverse Weather"
            else:
                reason = "Too many boats"
        else:
            color = "red1"
            if risk > num_boat:
                reason = "Adverse Weather"
            else:
                reason = "Too many boats"
        
        draw_indicator(gradient, w, int(risk_value), True)
        risk_value = (str(risk_value).zfill(2))

        frame.config(text = risk_value, bg=color, fg= "black", font=("Arial", 31))      
        rframe.config(text = reason, font=("Arial", 14, 'bold'))
        

def update_time(title):
        current_time = datetime.now().strftime("%A, %d/%m/%Y, %H:%M")
        name = current_time + ", Augusta:"
        title.config(text=name)
        root.after(60000,update_time,title)

def update_weather(l1,l2,r1,r2):
    while running:
        message, _, _ = get_weather(True)
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
            sunset = data["sys"]["sunset"]
            sunrise = data["sys"]["sunrise"]

            message = (
                f"Weather: {weather_description}\n"
                f"Temperature: {temperature}°C\n"
                f"Humidity: {humidity}%\n"
                f"Wind Speed: {wind_speed} m/s\n"
            )
        else:
            print(f"Error: {data['message']}")
        if call:
            return message, sunset, sunrise
        else:
            return weather_description, wind_speed
    except Exception as e:
        print(f"An error occurred: {e}")

def add_event(event_log):
    global event_running
    text = ""
    found = {}
    for cname in detected.keys():
        found[cname] = len(detected[cname])
        if detected[cname]:
            for i in range(len(detected[cname])):                
                detected[cname][i].update(label = names[np.argmax(detected[cname][i].tot_labels)])
                detected[cname][i].update(direction = direction[np.argmax(detected[cname][i].tot_direction)])
                lab = detected[cname][i].label             
                mov = detected[cname][i].direction

                if text == "":
                    text = cname + " : A " + lab + " is " + mov
                    actual = cname
                else:
                    if actual == cname:
                        text = text + " & A " + lab + " is " + mov
                        actual = cname
                    elif text[-1] == "\n":
                        text = text + cname + " : A " + lab + " is " + mov
                        actual = cname
        else:
            text = text + cname + " : No detection"
            actual = cname
        text = text + "\n"
    
    event_log.config(text = text)
    update_summary(found, text)
    
    event_running = False

def update_summary(found, text):
    summ = ""
    max_n = []
    _, sunset, sunrise = get_weather(True)
    for i in range(len(found.values())//2):
        key = list(found.keys())
        values = list(found.values())
        check_val = [values[i], values[i+2]]
        check_key = [key[i], key[i+2]]
        if check_val[1:] != check_val[:1]: 
            max = np.argmax(check_val)
            k = check_key[max]
        else:
            if int(time.time()) > sunset:
                k = check_key[-1] #dopo sunset
            elif int(time.time()) < sunrise:
                k = check_key[-1] #prima sunrise
            elif int(time.time()) > sunrise & int(time.time()) < sunset:
                k = check_key[0] #giorno
        max_n.append(k)

    for i in range(len(max_n)):
        if max_n[i] in text:
            start_index = text.find(max_n[i])
            end_index = text.find('\n', start_index)
            result = text[start_index:end_index].split(" : ")[-1]
            if result != "No detection":
                if summ == "":
                    summ = result.split(" : ")[-1]
                else:
                    summ = summ + " & " + result.split(" : ")[-1]
    
    summary.config(text = summ, font=("Montserrat", 22, 'bold'))


def add_frame(VFrame, IRFrame):
    global frame_running
    flag = False
    VISFr = None
    IRFr = None
    for cname in detected.keys():
        if detected[cname]:
            flag = True
            for boat in detected[cname]:
                if boat.cropped is not None:
                    img = boat.cropped
                    if "Camera Stream 1" in cname or "Camera Stream 2" in cname:
                        if VISFr is None:
                            VISFr = img
                        else:
                            img1 = VISFr
                            img2 = img
                            VISFr = cv2.vconcat([img1,img2])
                    else:
                        if IRFr is None:
                            IRFr = img
                        else:
                            img1 = IRFr
                            img2 = img
                            IRFr = cv2.vconcat([img1,img2])
        
    if not flag:
        IRFr = no_det
        VISFr = no_det
    Vimage = tk.PhotoImage(data=cv2.imencode(".ppm", VISFr)[1].tobytes())
    VFrame.configure(image = Vimage)
    VFrame.image = Vimage       
    update_scroll_region_vis()
    
    IRimage = tk.PhotoImage(data=cv2.imencode(".ppm", IRFr)[1].tobytes())
    IRFrame.configure(image =IRimage)
    IRFrame.image = IRimage
    update_scroll_region_ir()

    frame_running = False
    return

def resize_frame(bbox, frame):
    if bbox is not None:
        bbox = bbox.astype(int)
        x1,y1,x2,y2 = np.reshape(bbox, (4,))
        frame = frame[y1:y2, x1:x2]
    target_width, target_height = 200, 80 
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
        return 1 
    elif delta_x > 0:
        return 0


def detect_objects(frame,camera_name):

    bbox = []
    labels = []
    name = camera_name.split()[-1]
    if name == "1" or name == "2":
        model = modelVIS
    else:
        model = modelIR
    results = model(frame, device=0, imgsz=(320,352), verbose=False, iou=0.2, conf=0.4, agnostic_nms=True)

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
            print(f"Failed to open camera: {camera_address}. Retrying in 2 seconds...")
            time.sleep(2)
            continue

        while running:
            ret, frame = cap.read()
            
            if ret:
                with lock:  
                    frame, flag,bboxes, labels= detect_objects(frame, camera_name)  # Perform object detection on the frame

                if flag:
                    if detected[camera_name]:
                        Boat.remove_old(detected[camera_name])  
                    classes = frame.boxes.cls.cpu().numpy().astype(int)
                    if (len(labels) > 1):
                        classes = frame.boxes.cls.cpu().numpy().astype(int)
                        for i in range (len(bboxes)):
                            box = bboxes[i]
                            cl = classes[i]
                            box_norm = frame.boxes.xyxyn[i].cpu().numpy()
                            _, zone_y = calculate_box_center(box_norm)
                            centerx, centery = calculate_box_center(box)

                            if not detected[camera_name]:
                                id = "id_" + str(0)
                                detected[camera_name].append(Boat(id,[centerx,centery], datetime.timestamp(datetime.now()), 
                                                                  datetime.now().strftime("%H:%M:%S"), camera_name, 
                                                                  np.zeros(5).astype(int), np.zeros(2).astype(int), 
                                                                  zone = int(zone_y/0.2),  cropped = resize_frame(box, frame[0].orig_img)))
                            else:
                                pos, found = search_nearest([centerx,centery], camera_name)
                                if not found:
                                    id = ("id_"  + str(int(detected[camera_name][pos].id.split("_")[1]) + 1))
                                    detected[camera_name].append(Boat(id,[centerx,centery], datetime.timestamp(datetime.now()), 
                                                                      datetime.now().strftime("%H:%M:%S"), camera_name, 
                                                                      np.zeros(5).astype(int), np.zeros(2).astype(int), 
                                                                      zone = int(zone_y/0.2), cropped = resize_frame(box, frame[0].orig_img)))
                                else:
                                    oldx = detected[camera_name][pos].pos[0]
                                    oldy = detected[camera_name][pos].pos[1]
                                    dir = determine_movement_direction(centerx, centery, oldx, oldy)
                                    if Boat.check_id(detected[camera_name],camera_name,id):
                                        detected[camera_name][pos].update(pos=[centerx,centery], last_seen=datetime.timestamp(datetime.now()), 
                                                                          label_pos = cl, zone = int(zone_y/0.2), direction_pos = dir, 
                                                                          cropped = resize_frame(box, frame[0].orig_img))

                            if Boat.check_id(detected[camera_name],camera_name, id):
                                    pos = Boat.find_id(detected[camera_name], camera_name, id) 
                                    labels_array = detected[camera_name][pos].tot_labels
                                    index = np.argmax(labels_array)
                                    lab = labels_array[index]
                                    if lab >= 10:
                                        label = names[index]
                                    else:
                                        label = "boat"
                                    detected[camera_name][pos].update(label=label)

                            if Boat.check_id(detected[camera_name],camera_name, id):
                                pos = Boat.find_id(detected[camera_name],camera_name, id) 
                                label = names[np.argmax(detected[camera_name][pos].tot_labels)]


                            if Boat.check_id(detected[camera_name],camera_name, id):
                                pos = Boat.find_id(detected[camera_name], camera_name, id) 
                                direction_array = detected[camera_name][pos].tot_direction
                                index = np.argmax(direction_array)
                                mov = labels_array[index]
                                if mov >= 10:
                                    if mov: #vero approaching // falso leaving
                                        stat = "approaching"
                                    else:   
                                        stat = "leaving"
                                else:
                                    stat = "detected"
                                detected[camera_name][pos].update(direction=stat)
                        frame = frame.plot()
                    else:
                        box = frame[0].boxes.xyxy[0].cpu().numpy()
                        label = frame[0].names[int(frame[0].boxes.cls[0])]
                        centerx, centery = calculate_box_center(box)
                        box_norm = frame.boxes.xyxyn[0].cpu().numpy()
                        _, zone_y = calculate_box_center(box_norm)
                        cl = classes[-1]
                        
                        if not detected[camera_name]:
                            id = "id_" + str(0)
                            detected[camera_name].append(Boat(id,[centerx,centery],
                                                              datetime.timestamp(datetime.now()),
                                                              datetime.now().strftime("%H:%M:%S"),
                                                              camera_name, np.zeros(5).astype(int), np.zeros(2).astype(int),
                                                              zone = int(zone_y/0.2), cropped = resize_frame(box, frame[0].orig_img)))
                        else:
                            pos, found = search_nearest([centerx,centery], camera_name)
                            if not found:
                                id = ("id_"  + str(int(detected[camera_name][pos].id.split("_")[1]) + 1))
                                detected[camera_name].append(Boat(id,[centerx,centery], 
                                                                  datetime.timestamp(datetime.now()),
                                                                  datetime.now().strftime("%H:%M:%S"),
                                                                  camera_name,np.zeros(5).astype(int), np.zeros(2).astype(int),
                                                                  zone = int(zone_y/0.2), cropped = resize_frame(box, frame[0].orig_img)))
                            else:
                                oldx = detected[camera_name][pos].pos[0]
                                oldy = detected[camera_name][pos].pos[1]
                                dir = determine_movement_direction(centerx, centery, oldx, oldy)
                                if Boat.check_id(detected[camera_name],camera_name,id):
                                    detected[camera_name][pos].update(pos=[centerx,centery], last_seen=datetime.timestamp(datetime.now()), 
                                                                      label_pos = cl, zone = int(zone_y/0.2), direction_pos = dir, 
                                                                      cropped = resize_frame(box, frame[0].orig_img))

                        if Boat.check_id(detected[camera_name],camera_name, id):
                            pos = Boat.find_id(detected[camera_name], camera_name, id) 
                            labels_array = detected[camera_name][pos].tot_labels
                            index = np.argmax(labels_array)
                            lab = labels_array[index]
                            if lab >= 10:
                                label = names[index]
                            else:
                                label = "boat"

                        if Boat.check_id(detected[camera_name],camera_name, id):
                            pos = Boat.find_id(detected[camera_name], camera_name, id) 
                            direction_array = detected[camera_name][pos].tot_direction
                            index = np.argmax(direction_array)
                            mov = labels_array[index]
                            if mov >= 10:
                                if mov: #vero approaching // falso leaving
                                    stat = "approaching"
                                else:   
                                    stat = "leaving"
                            else:
                                stat = "detected"
                            detected[camera_name][pos].update(direction=stat)
                        frame = frame.plot()
                    #threading.Thread(target=add_frame, args=(VFrame,IRFrame),daemon=True).start()
                    with lock:
                        global frame_running
                        if not frame_running:
                            frame_running = True
                            threading.Timer(1, add_frame, args=(VFrame, IRFrame)).start()

                target_width, target_height = 320, 352  
                aspect_ratio = frame.shape[1] / frame.shape[0]
                if aspect_ratio > target_width / target_height:
                    target_height = int(target_width / aspect_ratio)
                else:
                    target_width = int(target_height * aspect_ratio)
                resized_frame = cv2.resize(frame, (target_width, target_height))
                image = tk.PhotoImage(data=cv2.imencode(".ppm", resized_frame)[1].tobytes())
                quadrant.configure(image=image)
                quadrant.image = image
                with lock:
                    global event_running
                    if not event_running:
                        event_running = True
                        root.after(1000,add_event,event_log)
 
            else:
                print(f"Failed to read frame from camera: {camera_address}. Retrying...")
                cap.release()
                break

        cap.release()



def create_gui(root):
    global q1_frame
    # # Main area divided into 4 quadrants
    q1_frame = tk.LabelFrame(root, text="Camera Stream 1:", width= 355, height= 180, labelanchor="n", 
                             font=font.Font(weight="bold"), bg=root.cget("bg"), foreground="#778DA9")
    q1_frame.pack_propagate(0)
    c1 = tk.Frame(q1_frame, bg=root.cget("bg"))
    c1.pack(side="top", padx=5, pady=5)
    quadrant_1 = tk.Label(c1, bg=root.cget("bg"))
    quadrant_1.pack(anchor="center")
    ToolTip(q1_frame, "First Visible Camera")


    q2_frame = tk.LabelFrame(root, text="Camera Stream 2:", width= 355, height= 180 ,labelanchor="n", 
                             font=font.Font(weight="bold"), bg=root.cget("bg"), foreground="#778DA9")
    q2_frame.pack_propagate(0)
    c2 = tk.Frame(q2_frame, bg=root.cget("bg"))
    c2.pack(side="top", padx=5, pady=5)
    quadrant_2 = tk.Label(c2,bg=root.cget("bg"))
    quadrant_2.pack(anchor="center")
    ToolTip(q2_frame, "Second Visible Camera")


    q3_frame = tk.LabelFrame(root, text="Camera Stream 3:", width= 355, height= 252, labelanchor="n", 
                             font=font.Font(weight="bold"), bg=root.cget("bg"), foreground="#778DA9")
    q3_frame.pack_propagate(0)
    c3 = tk.Frame(q3_frame, bg=root.cget("bg"))
    c3.pack(side="top", padx=5, pady=5)
    quadrant_3 = tk.Label(c3, bg=root.cget("bg"))
    quadrant_3.pack(anchor="center")
    ToolTip(q3_frame, "First IR Camera")

    q4_frame = tk.LabelFrame(root, text="Camera Stream 4:", width= 355, height= 252, labelanchor="n", 
                             font=font.Font(weight="bold"), bg=root.cget("bg"), foreground="#778DA9")
    q4_frame.pack_propagate(0)
    c4 = tk.Frame(q4_frame, bg=root.cget("bg"))
    c4.pack(side="top", padx=5, pady=5)
    quadrant_4 = tk.Label(c4, bg=root.cget("bg"))
    quadrant_4.pack(anchor="center")
    ToolTip(q4_frame, "Second IR Camera")


    # Event log frame
    event_log_frame = tk.LabelFrame(root, text="Event Log:", width=380, height=180,  labelanchor="n", 
                                    font=font.Font(weight="bold"), bg=root.cget("bg"), foreground="#778DA9")
    event_log_frame.pack_propagate(0)
    columns = tk.Frame(event_log_frame, bg=root.cget("bg"))
    columns.pack(side="top", padx=5, pady=5, anchor="nw")
    elog = tk.Label(columns, wraplength=350, bg=root.cget("bg"), justify=tk.LEFT)
    elog.pack(anchor="w")
    ToolTip(event_log_frame, "Log of event for each camera")

    # Summary frame
    global summary
    summary_frame = tk.LabelFrame(root, text="Summary Log:", width=380, height=140,  labelanchor="n", 
                                  font=font.Font(weight="bold"), bg=root.cget("bg"), foreground="#778DA9")
    summary_frame.pack_propagate(0)
    column = tk.Frame(summary_frame, bg=root.cget("bg"))
    column.pack(side="top", padx=5, pady=5, anchor="center")
    summary = tk.Label(column, wraplength=400, bg=root.cget("bg"), justify=tk.LEFT, anchor="center")
    summary.pack(anchor="center")
    ToolTip(summary_frame, "Summary of the situation of the port")


    # Additional information in two columns
    additional_info_frame = tk.LabelFrame(root, text="Weather in Augusta:", width= 339, height= 83, labelanchor="n", 
                                          font=font.Font(weight="bold"), bg=root.cget("bg"), foreground="#778DA9")
    additional_info_frame.pack_propagate(0)
    left_column = tk.Frame(additional_info_frame, bg=root.cget("bg"))
    left_column.pack(side="left", padx=5, pady=5)
    right_column = tk.Frame(additional_info_frame, bg=root.cget("bg"))
    right_column.pack(side="right", padx=5, pady=5)
    left_label1 = tk.Label(left_column, bg=root.cget("bg"))
    left_label1.pack(anchor="w")
    left_label2 = tk.Label(left_column, bg=root.cget("bg"))
    left_label2.pack(anchor="w")
    right_label1 = tk.Label(right_column, bg=root.cget("bg"))
    right_label1.pack(anchor="e")
    right_label2 = tk.Label(right_column, bg=root.cget("bg"))
    right_label2.pack(anchor="e")
    ToolTip(additional_info_frame, "Weather information")


    # Visibile and IR Cam frames
    global VFrame, IRFrame, scrollbarVIS, scrollbarIR
    visible_frame = tk.LabelFrame(root, text="Visibile Cam:", width=200, height=225,  pady=2, 
                                  labelanchor="n", font=font.Font(weight="bold"), bg=root.cget("bg"), 
                                  foreground="#778DA9")
    #visible_frame.pack_propagate(0)
    scrollbarVIS = tk.Canvas(visible_frame,bg=root.cget("bg"), highlightbackground=root.cget("bg"), 
                             width=202, height=225) #scrollregion=(0,0,0,500)
    scrollbarVIS.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    y_scrollbar = ttk.Scrollbar(visible_frame, orient=tk.VERTICAL, command=scrollbarVIS.yview, 
                                style="Vertical.TScrollbar")
    y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbarVIS.configure(yscrollcommand=y_scrollbar.set)
    frame = tk.Frame(scrollbarVIS,bg=root.cget("bg"))
    frame.pack(side="top", padx=5, pady=5)
    VFrame = tk.Label(frame, bg=root.cget("bg"))
    VFrame.pack(anchor="center")
    scrollbarVIS.create_window((0, 0), window=frame, anchor="nw")
    ToolTip(visible_frame, "Frame for boat visualization")

    IR_frame = tk.LabelFrame(root, text="IR Cam:", width=200, height=225, padx=2, pady=2, 
                             labelanchor="n", font=font.Font(weight="bold"), bg=root.cget("bg"), 
                             foreground="#778DA9")
    #IR_frame.pack_propagate(0)
    scrollbarIR = tk.Canvas(IR_frame,bg=root.cget("bg"), highlightbackground=root.cget("bg"), 
                            width=202,height=225)
    scrollbarIR.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    y_scrollbar2 = ttk.Scrollbar(IR_frame, orient=tk.VERTICAL, command=scrollbarIR.yview, 
                                 style="Vertical.TScrollbar")
    y_scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbarIR.configure(yscrollcommand=y_scrollbar2.set)
    frame2 = tk.Frame(scrollbarIR, bg=root.cget("bg"))
    frame2.pack(side="top", padx=5, pady=5)
    IRFrame = tk.Label(frame2, bg=root.cget("bg"))
    IRFrame.pack(anchor="center")
    scrollbarIR.create_window((0, 0), window=frame2, anchor="nw")
    ToolTip(IR_frame, "Frame for boat visualization")

    #Risk indicator
    Risk_frame = tk.LabelFrame(root, text="Alert indicator:", width=280, height=95, labelanchor="n", 
                               font=font.Font(weight="bold"), bg=root.cget("bg"), foreground="#778DA9")
    Risk_frame.pack_propagate(0)
    gradient = tk.Canvas(Risk_frame, width=255*2, height=100, bg=root.cget("bg"), highlightbackground=root.cget("bg"))
    draw_gradient(gradient)
    gradient.pack(side="left",padx=5,pady=2)
    w,h = resize_canvas(gradient)
    draw_indicator(gradient, w, 100, False)
    gradient.config(width=w, height=70)
    frame3 = tk.Frame(Risk_frame, bg=root.cget("bg"))
    frame3.pack()
    RiskFrame = tk.Label(frame3, anchor="center", bg=root.cget("bg"))
    RiskFrame.pack(side="right",padx=5,pady=2)
    ToolTip(Risk_frame, "Alert indicator of the current situation")

    Risk_reason_frame = tk.LabelFrame(root, text="Reason:", width=190, height=72, labelanchor="n", 
                                      font=font.Font(weight="bold"), bg=root.cget("bg"), foreground="#778DA9")
    Risk_reason_frame.pack_propagate(0)
    frame4 = tk.Frame(Risk_reason_frame, bg=root.cget("bg"))
    frame4.pack(side="top", padx=2, pady=2)
    RiskReason = tk.Label(frame4, anchor="center", bg=root.cget("bg"))
    RiskReason.pack()

    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)
    root.grid_rowconfigure(2, weight=1)
    root.grid_columnconfigure(3, weight=1)
    root.grid_rowconfigure(3, weight=1)

    q1_frame.grid(row=0, column=0, padx=5, sticky="nsew")
    q2_frame.grid(row=0, column=1, padx=5, sticky="nsew")
    q3_frame.grid(row=1, column=0, padx=5, sticky="nsew")
    q4_frame.grid(row=1, column=1, padx=5, sticky="nsew")

    additional_info_frame.grid(row=2, column=0, columnspan=1, padx=5, pady=5, sticky="w")
    event_log_frame.grid(row=0, column=2, rowspan=1, columnspan=2, padx=5, pady=5, sticky="n")
    visible_frame.grid(row=1, column=2,rowspan=1, columnspan=1, padx=0, pady=5, sticky="nw")
    IR_frame.grid(row=1, column=3, rowspan=1 ,padx=0, pady=5, sticky="ne")
    summary_frame.grid(row=2, column=2, rowspan=1, columnspan=2, padx=5, pady=5, sticky="nsew")
    Risk_frame.grid(row=2, column=1, columnspan=1, padx=5, pady=5, sticky="n")
    Risk_reason_frame.grid(row=2, column=1, columnspan=1, padx=5, pady=5, sticky="s")

    
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

    global threading
    for camera_name, camera_address in mareforte.items():
        quadrant = locals()[f"quadrant_{camera_name.split()[-1]}"]
        thread = threading.Thread(target=display_camera_stream, args=(camera_address, quadrant, 
                                                                      elog, camera_name), daemon=True)
        threads.append(thread)
    threads.append(threading.Thread(target=update_weather, args=(left_label1, left_label2, right_label1, 
                                                                 right_label2), daemon=True))
    threads.append(threading.Thread(target=update_time, args=(additional_info_frame,), daemon=True))
    threads.append(threading.Thread(target=risk_factor, args=(RiskFrame,RiskReason, gradient, w), daemon=True))
    for thread in threads:
        thread.start()


def on_closing():
    global running, threading
    running = False
    time.sleep(2)
    root.destroy() 
    sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    root.configure(bg='#0D1B2A')
    root.title("Camera Streams GUI")
    root.option_add("*Label.foreground", "#E0E1DD")
    root.geometry("1200x800")
    icon = tk.PhotoImage(file="utils/icon.png")
    root.iconphoto(True, icon)
    style.configure("Vertical.TScrollbar", gripcount=0,
                background="Green", darkcolor="DarkGreen", lightcolor="LightGreen",
                troughcolor="gray", bordercolor="blue", arrowcolor="white")
    style = ttk.Style()
    style.theme_use('clam')
    create_gui(root)

    
    root.protocol("WM_DELETE_WINDOW", on_closing)  

    root.mainloop()

    for thread in threading.enumerate():
        if thread != threading.current_thread():
            thread.join()    
