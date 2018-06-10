import cv2
import face_recognition

'''
This server:
    Input: Camera frame
    Output: Relative locations for each face, with [(tr_x, tr_y, bl_x, bl_y)]

x1,y1 ------
|          |
|          |
|          |
--------x2,y2
'''


class FaceTrackServer(object):

    faces = []
    face_locations = []
    face_relative_locations = []
    cam_h = None
    cam_w = None
    camera_address = None

    def __init__(self, down_scale_factor=0.25):
        assert 0 <= down_scale_factor <= 1
        self.down_scale_factor = down_scale_factor

    def get_cam_info(self):
        return {'camera': {'width': self.cam_w, 'height': self.cam_h, 'address': self.camera_address}}

    def reset(self):
        self.face_relative_locations = []
        self.face_locations = []
        self.faces = []

    def process(self, frame):
        self.reset()
        self.cam_h, self.cam_w, _ = frame.shape
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.down_scale_factor, fy=self.down_scale_factor)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        # Display the results
        for y1_sm, x2_sm, y2_sm, x1_sm in self.face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            x1 = int(x1_sm / self.down_scale_factor)
            x2 = int(x2_sm / self.down_scale_factor)
            y1 = int(y1_sm / self.down_scale_factor)
            y2 = int(y2_sm / self.down_scale_factor)

            x1_rltv = x1 / self.cam_w
            x2_rltv = x2 / self.cam_w
            y1_rltv = y1 / self.cam_h
            y2_rltv = y2 / self.cam_h

            _face_area = frame[x1:x2, y1:y2, :]
            if _face_area.size == 0:
                continue
            self.faces.append(_face_area)
            self.face_relative_locations.append([x1_rltv, y1_rltv, x2_rltv, y2_rltv])
            # cv2.imshow('faces', frame[y1:y2, x1:x2, :])
            # cv2.waitKey(0)
        print('[FaceTracker Server] Found {} faces!'.format(len(self.faces)))
        return self.faces

    def get_faces_loc(self, relative=True):
        if relative:
            return self.face_relative_locations
        else:
            return self.face_locations

    def get_faces(self):
        return self.faces