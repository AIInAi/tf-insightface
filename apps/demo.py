import cv2
import numpy as np
from models import face_track_server, face_describer_server, face_db, camera_server
from configs import configs


class Demo(camera_server.CameraServer):

    def __init__(self, *args, **kwargs):
        super(Demo, self).__init__(*args, **kwargs)
        self.face_tracker = face_track_server.FaceTrackServer()
        self.face_describer = face_describer_server.FDServer(
            model_fp=configs.face_describer_model_fp,
            input_tensor_names=configs.face_describer_input_tensor_names,
            output_tensor_names=configs.face_describer_output_tensor_names,
            device=configs.face_describer_device)
        self.face_db = face_db.Model()

    def processs(self, frame):
        # Step1. Find and track face (frame ---> [Face_Tracker] ---> Faces Loactions)
        self.face_tracker.process(frame)
        _faces = self.face_tracker.get_faces()

        # Uncomment below to visualize face
        # _faces_loc = self.face_tracker.get_faces_loc()
        # self._viz_faces(_faces_loc, frame)

        # Step2. For each face, get the cropped face area, feeding it to face describer (insightface) to get 512-D Feature Embedding
        _drop_out_rate = 0.1
        _face_descriptions = []
        _num_faces = len(_faces)
        if _num_faces == 0:
            return
        for _face in _faces:
            _face_resize = cv2.resize(_face, configs.face_describer_tensor_shape)
            _data_feed = [np.expand_dims(_face_resize, axis=0), _drop_out_rate]
            _face_description = self.face_describer.inference(_data_feed)[0][0]
            _face_descriptions.append(_face_description)

            # Step3. For each face, check whether there are similar faces and if not save it to db.
            # Below naive and verbose implementation is to tutor you how this work
            _similar_faces = self.face_db.get_similar_faces(_face_description)
            if len(_similar_faces) == 0 or len(self.face_db.faces) == 0:
                self.face_db.add_face(face_img=_face, face_description=_face_description)
        print('[Demo] -----------------------------------------------------------')

    def _viz_faces(self, faces_loc, frame):
        for _face_loc in faces_loc:
            x1 = int(_face_loc[0] * self.face_tracker.cam_w)
            y1 = int(_face_loc[1] * self.face_tracker.cam_h)
            x2 = int(_face_loc[2] * self.face_tracker.cam_w)
            y2 = int(_face_loc[3] * self.face_tracker.cam_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('faces', frame)
        cv2.waitKey(1)

demo = Demo(camera_address=0)
demo.run()
