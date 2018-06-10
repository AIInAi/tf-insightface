import cv2

'''
This server:
    Input: Camera address
    Process: Start a camera thread
    Output: Run an overridden process function for each frame
'''


class CameraServer(object):

    camera_address = None
    cam = None
    in_progress = False

    def __init__(self, camera_address):
        self.camera_address = camera_address

    def get_status(self):
        return self.in_progress

    # Must be overridden
    def processs(self, frame):
        raise NotImplementedError

    def run(self):
        print('[Camera Server] Camera is initializing ...')
        if self.camera_address is not None:
            self.cam = cv2.VideoCapture(self.camera_address)
        else:
            print('[Camera Server] Camera is not available!')
            return

        while True:
            self.in_progress = True

            # Grab a single frame of video
            ret, frame = self.cam.read()
            self.processs(frame)
        self.in_progress = False
