from datetime import datetime
import os


filename = datetime.now().strftime("%H_%M_%S") + "_log.csv"

def save_log(data):
    directory = "logs"
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory,filename)
    file_exists = os.path.exists(path)
    fieldnames = "Camera,Label,Time,Direction of travel"
    with open(path, "a") as file:
        if not file_exists:
            file.write(fieldnames)
            file.write("\n")
        file.write(data)
        file.write("\n")


class Boat:
    def __init__(self,id,pos,last_seen, arrival, camera,
                 tot_labels, tot_direction, label="boat", 
                 direction="detected", zone=None, cropped=None):
        self.id = id
        self.pos = pos
        self.last_seen = last_seen
        self.arrival = arrival
        self.camera = camera
        self.tot_labels = tot_labels
        self.tot_direction = tot_direction
        self.label = label
        self.direction = direction
        self.zone = zone
        self.cropped = cropped

    def update(self, id=None, pos=None, arrival=None, last_seen=None, camera=None, label_pos=None, direction_pos=None, label=None, direction=None, zone=None, cropped=None):
        if id is not None:
            self.id = id
        if pos is not None:
            self.pos = pos
        if arrival is not None:
            self.arrival = arrival
        if last_seen is not None:
            self.last_seen = last_seen
        if camera is not None:
            self.camera = camera
        if label_pos is not None:
            self.tot_labels[label_pos] += 1
        if direction_pos is not None:
            self.tot_direction[direction_pos] += 1
        if label is not None:
            self.label = label
        if direction is not None:
            self.direction = direction
        if zone is not None:
            self.zone = zone
        if cropped is not None:
            self.cropped = cropped

    def find_id(list,camera_name, id):
        for index, boat in enumerate(list):
            if boat.id == id and boat.camera == camera_name:
                return index
        return False

    @staticmethod
    def check_id(list,camera_name, id):
        return any(boat.id == id and boat.camera == camera_name for boat in list)

    def remove_old(list):
        now = datetime.timestamp(datetime.now())
        for boat in list:
            if now - boat.last_seen > 10:
                #print(f"Removing: {boat}")
                data = boat.camera + "," + boat.label + "," + boat.arrival + "," + boat.direction
                save_log(data)
                list.remove(boat)


    def __repr__(self):
            return (f"ID: {self.id}, Time: {self.last_seen}, Position : {self.pos}, Camera: {self.camera}, LabelArray: {self.tot_labels}, DirectionArray: {self.tot_direction}, Label: {self.label}, Direction: {self.direction}, Arrival Time: {self.arrival}\n")