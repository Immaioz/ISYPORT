import tkinter as tk
import cv2
import threading
import time
import numpy as np
import requests
from datetime import datetime
from ultralytics import YOLO

lock = threading.Lock()

# Load YOLO model
model = YOLO('/home/antonino/Università/porto/train/ModelloTotale/weights/best.pt')

#classes = list(model.names.values())

# Global flag to control threads
running = True

global info
global old_centerx
global status

info = {
    "Camera Stream 1": None,
    "Camera Stream 2": None,
    "Camera Stream 3": None,
    "Camera Stream 4": None
}

old_centerx = {
    "Camera Stream 1": None,
    "Camera Stream 2": None,
    "Camera Stream 3": None,
    "Camera Stream 4": None
}

status = {
    "Camera Stream 1": None,
    "Camera Stream 2": None,
    "Camera Stream 3": None,
    "Camera Stream 4": None
}


def get_weather():
    api_key = "916923f32ca072e53d4f02822dbc6968"
    city = "Augusta, IT"  # Replace with the desired city name
    units = "metric"   # Use "imperial" for Fahrenheit, "metric" for Celsius, or "standard" for Kelvin
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units={units}&appid={api_key}"
    current_time = datetime.now().strftime("%A, %d/%m/%Y, %H:%M:%S")
    try:
        response = requests.get(base_url)
        data = response.json()
        
        if response.status_code == 200:
            weather_description = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            
            message = (
                f"{current_time}, Augusta:\n"
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

def update_weather(title, l1,l2,r1,r2):
    while True:
        message = get_weather()
        mex = message.split("\n")
        title.config(text=mex[0])
        l1.config(text=mex[1])
        l2.config(text=mex[2])
        r1.config(text=mex[3])
        r2.config(text=mex[4])
        time.sleep(1)

def calculate_box_center(corners):
    bbox = np.reshape(corners, (4,))
    xmin, ymin, xmax, ymax = bbox
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    return center_x, center_y

def determine_movement_direction(x1, x2):
    delta_x = x2 - x1
    if delta_x > 0:
        return True
    elif delta_x < 0:
        return False

def add_event(event_log, messages, image):
    text = ""
    for key, value in messages.items():
        if value == None:
            text = text + key + " : No detection " + "\n"
        else:
            text = text + key + " : " + value + "\n"
    event_log.config(text= text)
    # event_log.image_create(tk.END, image=image)
    # event_log.config(state=tk.DISABLED)  # Disable editing

def detect_objects(frame):

    results = model(frame, device=0, imgsz=(800,480), verbose=False)

    if len(results[0].boxes) != 0:
        return results[0], True

    return frame, False


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
                    frame, flag = detect_objects(frame)  # Perform object detection on the frame

                if flag:
                    box = frame[0].boxes.xyxy[0].cpu().numpy()
                    label = frame[0].names[int(frame[0].boxes.cls[0])]
                    centerx, centery = calculate_box_center(box)
                    if old_centerx[camera_name] == None:
                        old_centerx[camera_name] = centerx
                        state = "detected"
                    else:
                        old = old_centerx[camera_name]
                        status[camera_name] = determine_movement_direction(centerx, old)
                        if status[camera_name] == False:
                            state = "leaving"
                        else:
                            state = "approaching"
        
                    message = "A " + label + " is " + state
                    
                    info[camera_name] =  message
                    frame = frame.plot()
                else:
                    info[camera_name] = None

                target_width, target_height = 350, 300  # Adjust as needed
                aspect_ratio = frame.shape[1] / frame.shape[0]
                if aspect_ratio > target_width / target_height:
                    target_height = int(target_width / aspect_ratio)
                else:
                    target_width = int(target_height * aspect_ratio)
                resized_frame = cv2.resize(frame, (target_width, target_height))
                image = tk.PhotoImage(data=cv2.imencode(".ppm", resized_frame)[1].tobytes())
                quadrant.configure(image=image)
                quadrant.image = image
                add_event(event_log, info, image)
            else:
                print(f"Failed to read frame from camera: {camera_address}. Retrying...")
                cap.release()
                break

        cap.release()

def create_gui(root):
    # Main area divided into 4 quadrants
    quadrant_1 = tk.Label(root, bg="grey", width=60, height=30)  # Larger size for higher resolution
    quadrant_2 = tk.Label(root, bg="grey", width=60, height=30)  # Larger size for higher resolution
    quadrant_3 = tk.Label(root, bg="grey", width=60, height=30)  # Larger size for higher resolution
    quadrant_4 = tk.Label(root, bg="grey", width=60, height=30)  # Larger size for higher resolution

    # Event log frame
    event_log_frame = tk.LabelFrame(root, text="Event Log", width=30, height=30)
    columns = tk.Frame(event_log_frame)
    columns.pack(side="top", padx=5, pady=5)
    elog = tk.Label(columns)
    elog.pack(anchor="ne")


    # Additional information in two columns
    additional_info_frame = tk.LabelFrame(root, text="Additional Information", width=60)
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

    quadrant_1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    quadrant_2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
    quadrant_3.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
    quadrant_4.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
    event_log_frame.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="n")
    additional_info_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
    
    cameras = {
        "Camera Stream 1": "Video/Cam1.mp4",
        "Camera Stream 2": "Video/Cam2.mp4",
        "Camera Stream 3": "Video/Cam3.mp4",
        "Camera Stream 4": "Video/Cam4.mp4"
    }

    for camera_name, camera_address in cameras.items():
        quadrant = locals()[f"quadrant_{camera_name.split()[-1]}"]
        threading.Thread(target=display_camera_stream, args=(camera_address, quadrant, elog, camera_name), daemon=True).start()
    threading.Thread(target=update_weather, args=(additional_info_frame,left_label1, left_label2, right_label1, right_label2), daemon=True).start()



def on_closing():
    global running
    running = False  # Set the global flag to stop threads
    root.destroy()   # Destroy the main GUI window

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Camera Streams GUI")
    root.geometry("1200x800")

    create_gui(root)

    root.protocol("WM_DELETE_WINDOW", on_closing)  # Intercept window close event

    root.mainloop()

    # After the main loop, wait for threads to finish
    for thread in threading.enumerate():
        if thread != threading.current_thread():
            thread.join()
