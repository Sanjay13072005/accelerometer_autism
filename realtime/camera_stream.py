import cv2

class CameraStream:
    def __init__(self, source=0):
        """
        source:
          0              → Laptop webcam
          http://IP/...  → IP Webcam
        """
        self.cap = cv2.VideoCapture(source)

    def read(self):
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
