import pandas as pd
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
import numpy as np
sys.path.append('../')
from utils import get_bounding_box_width, get_center_of_box

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('boundingbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'boundingbox': x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_objects_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        tracks = {
            "players" : [],
            "referees" : [],
            "ball" : []
        }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inv = {v:k for k,v in class_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = class_names_inv["player"]

            detection_with_tracking = self.tracker.update_with_detections(detection_supervision)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracking:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if class_id == class_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"boundingbox": bounding_box}

                if class_id == class_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"boundingbox": bounding_box}

            for frame_detection in detection_supervision:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"boundingbox": bounding_box}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bounding_box, color, track_id=None):
        y2 = int(bounding_box[3])
        x_center, _ = get_center_of_box(bounding_box)
        width = get_bounding_box_width(bounding_box)
        
        cv2.ellipse(
            img=frame,
            center=(x_center, y2),
            axes=(int(width), int(0.4*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=245,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rect_width = 40
        rect_height = 20
        x1_rect = x_center - rect_width//2
        x2_rect = x_center + rect_width//2
        y1_rect = (y2 - rect_height//2) + 15
        y2_rect = (y2 + rect_height//2) + 15

        if track_id is not None:
            cv2.rectangle(
                img=frame,
                pt1=(int(x1_rect), int(y1_rect)),
                pt2=(int(x2_rect), int(y2_rect)),
                color=color,
                thickness=cv2.FILLED
            )
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                img=frame,
                text=f"{track_id}",
                org=(int(x1_rect), int(y1_rect + 15)),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.6,
                color=(0, 0, 0),
                thickness=2
            )

        return frame
    
    def draw_triangle(self, frame, bounding_box, color):
        y = int(bounding_box[1])
        x, _ = get_center_of_box(bounding_box)

        triangle_pts = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_pts], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, vid_frames, tracks):
        output_vid_frames = []
        
        for frame_num, frame in enumerate(vid_frames):
            frame = frame.copy()

            players_dict = tracks["players"][frame_num]
            referees_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for track_id, player in players_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["boundingbox"], color, track_id)
            
            for _, referee in referees_dict.items():
                frame = self.draw_ellipse(frame, referee["boundingbox"], (0,255,255))
            
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["boundingbox"], (0,255,0))

            output_vid_frames.append(frame)

        return output_vid_frames