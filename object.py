#C:\miniconda\Conda_install\AI\Object_Detection\Object_detection
from ultralytics import YOLO
import cv2
import numpy as np
import time

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point2) - np.array(point1))

def main():
    model = YOLO('yolov8n.pt')  
    video_path = 'input_video.mp4'
    cap = cv2.VideoCapture(video_path)

    object_positions = {}  
    object_speeds = {}     
    movement_threshold = 2
    frame_time = time.time()


    target_classes = ['person', 'sports ball']
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1080, 720))
        current_frame_time = time.time()
        time_diff = current_frame_time - frame_time
        frame_time = current_frame_time

        results = model.track(frame, persist=True, conf=0.5, iou=0.5)
        moving, stationary = set(), set()

        for result in results:
            boxes = result.boxes.xywh
            ids = result.boxes.id
            class_ids = result.boxes.cls

            if ids is not None:
                for i, box in enumerate(boxes):
                    obj_id = int(ids[i])
                    class_id = int(class_ids[i])
                    class_name = model.names[class_id]

                    if class_name not in target_classes:
                        continue

                    center_x, center_y, w, h = box

                    if obj_id in object_positions:
                        prev = object_positions[obj_id]
                        curr = (center_x, center_y)
                        distance = calculate_distance(prev, curr)

                        if distance > movement_threshold:
                            moving.add(obj_id)
                        else:
                            stationary.add(obj_id)

                        distance_m = distance * 0.05
                        if time_diff > 0:
                            speed = distance_m / time_diff
                            object_speeds[obj_id] = speed

                            cv2.putText(
                                frame, f"{class_name} ID:{obj_id} {speed:.2f} m/s",
                                (int(center_x - w/2), int(center_y - h/2 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                            )

                    object_positions[obj_id] = (center_x, center_y)

        cv2.putText(frame, f"Moving: {len(moving)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Stationary: {len(stationary)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        frame_ = results[0].plot()
        cv2.imshow("Football Match Tracking", frame_)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
