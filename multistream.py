import cv2
from ultralytics import YOLO
import numpy as np
#import window_script
# Load the YOLOv8 model
# model = YOLO(r'D:\ExportVideo\Images\runs\detect\train\weights\best.pt')
model = YOLO('/home/antonino/Università/porto/train/ModelloTotale/weights/best.pt')
# Open the video file
video_path = "test1.mp4"
#cap = cv2.VideoCapture(video_path)

video_files = ['Video/Cam1.mp4', 'Video/Cam2.mp4', 'Video/Cam3.mp4', 'Video/Cam4.mp4']
caps = [cv2.VideoCapture(file) for file in video_files]

old_centerx = float('inf')

cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing


def calculate_box_center(corners):
    bbox = np.reshape(corners, (4,))
    xmin, ymin, xmax, ymax = bbox
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    return center_x, center_y

leaving = 0
approaching = 0
fr = 0
state = ""
frames = []

# Set the window size to a specific width and height
cv2.resizeWindow('Demo', 1080, 1024)  # Set width=800 and height=600

# Loop through the video frames

# Check if video files are opened successfully
for cap in caps:
    if not cap.isOpened():
        print(f'Failed to open {cap}')
        exit()

while True:
    # Read frames from each video
    for cap in caps:
        success, frame = cap.read()
        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, verbose=False)
            # Resize the frame to fit the subwindow (adjust the size as needed)
            #frame = cv2.resize(frame, height=512)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            boh = results[0].boxes
            frame_height, frame_width, _ = frame.shape
            font_scale = min(frame_width, frame_height) / 500.0
            #print("Altezza", frame_height)
            #print("Larghezza", frame_width)
            if len(boh) != 0:
                label = results[0].names[int(boh.cls[0])]
                box = results[0].boxes.xyxy[0].cpu().numpy()
               # print(box)
                centerx, centery = calculate_box_center(box)

                # Draw a point at the center on the image
                center_point = (int(centerx), int(centery))
                #print(center_point)
                cv2.circle(frame, center_point, 50, (0, 255, 0), -1)

                if old_centerx == float('inf'):
                    old_centerx = centerx
                else:
                    fr += 1
                    if centerx > old_centerx:
                        leaving +=1
                    else:
                        approaching += 1


                if fr == 50:
                    if leaving > approaching:
                        state = "leaving"
                    else:
                        state = "approaching"


                if state == "":
                    info = label
                else:
                    info = "A " + label + " is " + state

                #window_script.create_window(info)
                text_size, _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
                text_x = int((frame_width - text_size[0]) / 2)
                text_y = int(frame_height - (frame_height * 0.05))


                cv2.putText(frame, info, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 255), 1)
            frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
            frames.append(frame)

    # Display frames in subwindows
    # Concatenate frames horizontally for the first two videos
    top_row = cv2.hconcat([frames[0], frames[1]])

    # Concatenate frames horizontally for the second two videos
    bottom_row = cv2.hconcat([frames[2], frames[3]])

    # Concatenate top and bottom rows vertically
    final_frame = cv2.vconcat([top_row, bottom_row])



    cv2.imshow('Demo', final_frame)

    # Clear the list of frames
    frames.clear()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


for cap in caps:
    cap.release()
cv2.destroyAllWindows()



