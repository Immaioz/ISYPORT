import tkinter as tk
import cv2
import threading
import time
import numpy as np

# Global flag to control threads
running = True

def display_camera_stream(camera_address, quadrant):
    while running:
        cap = cv2.VideoCapture(camera_address)

        # Set resolution to 3840x2160
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

        if not cap.isOpened():
            print(f"Failed to open camera: {camera_address}. Retrying in 5 seconds...")
            time.sleep(5)  # Wait for 5 seconds before retrying
            continue

        while running:  # Check the global flag
            ret, frame = cap.read()
            if ret:
                # Calculate target dimensions while maintaining aspect ratio
                target_width, target_height = 800, 450  # Adjust as needed
                aspect_ratio = frame.shape[1] / frame.shape[0]
                if aspect_ratio > target_width / target_height:
                    target_height = int(target_width / aspect_ratio)
                else:
                    target_width = int(target_height * aspect_ratio)

                # Resize the frame
                resized_frame = cv2.resize(frame, (target_width, target_height))

                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                image = tk.PhotoImage(data=cv2.imencode(".ppm", frame_rgb)[1].tobytes())

                quadrant.configure(image=image)
                quadrant.image = image
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

    # Event log
    event_log = tk.Text(root, width=40, height=20)
    event_log.insert(tk.END, "Event Log:\n")
    event_log.insert(tk.END, "Event 1\n")
    event_log.insert(tk.END, "Event 2\n")
    event_log.insert(tk.END, "Event 3\n")
    event_log.insert(tk.END, "Event 4\n")
    event_log.insert(tk.END, "Event 5\n")
    event_log.config(state=tk.DISABLED)

    # Bottom space for additional information
    additional_info = tk.Label(root, text="Additional Information", bg="white", width=100, height=5)

    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)

    quadrant_1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    quadrant_2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
    quadrant_3.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
    quadrant_4.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
    event_log.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="nsew")
    additional_info.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

    cameras = {
        "Camera Stream 1": "Video/Cam1.mp4",
        "Camera Stream 2": "Video/Cam2.mp4",
	"Camera Stream 3": "Video/Cam3.mp4",
	"Camera Stream 4": "Video/Cam4.mp4"
    }

    for camera_name, camera_address in cameras.items():
        quadrant = locals()[f"quadrant_{camera_name.split()[-1]}"]
        threading.Thread(target=display_camera_stream, args=(camera_address, quadrant), daemon=True).start()

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
