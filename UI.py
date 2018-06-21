import sys
import time
import serial

import cv2
import numpy as np
import numpy.linalg as la
import cubicspline as cs
import goodfeature as gf

from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import*
import matplotlib.pyplot as plt

class Capture(QtCore.QThread):
    changePixmap = QtCore.pyqtSignal(QImage)
    
    def __init__(self, ComNumber):
        QtCore.QThread.__init__(self)

        self.width = 960
        self.height = 720
        self.K = np.matrix([[756.78768558,   0.        , 629.89805344],
                   [  0.        , 756.86336981, 345.49169401],
                   [  0.        ,   0.        ,   1.        ]])
        
        

        self.ser = serial.Serial(ComNumber, 115200, timeout = 0.01)
        if(self.ser.isOpen() == True):
            self.ser.close()

        self.ser.open()

    ### Warping ###
    def cos(self, angle):
        return np.cos(np.deg2rad(angle))

    def sin(self, angle):
        return np.sin(np.deg2rad(angle))

    def findZ(self, angle):
        R = 13.5
        if angle <= 90:
            return R
        elif angle > 90 and angle <= 135:
            return R*self.cos(angle - 90)
        else:
            return R*self.cos(180 - angle)

    def findHomography(self, r, p, y, t):
    
        Rx = np.matrix([[   self.cos(r), -self.sin(r), 0.],
                        [   self.sin(r),  self.cos(r), 0.],
                        [            0.,           0., 1.]])
    
        Ry = np.matrix([[1.,            0.,           0.],
                        [0.,   self.cos(p), -self.sin(p)], 
                        [0.,   self.sin(p),  self.cos(p)]])


        Rz = np.matrix([[ self.cos(y), 0., self.sin(y)],
                        [          0., 1.,          0.],
                        [-self.sin(y), 0., self.cos(y)]])

        R = Rx*Ry*Rz
        n = [0., 0., 1.]
        d = 50
    
        H = self.K*(R + t*np.transpose(n)/d)*la.inv(self.K)
        return(H)

    ### Flow ###
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 960)
        cap.set(4, 720)
        while 1:
            ret, frame = cap.read()
            
            self.ser.write(b'1')
            txt = self.ser.readline().decode('utf-8')
            values = txt.split(',')
            theta = float(values[0])%180
            roll = float(values[1])
            pitch = float(values[2])
            dz = self.findZ(theta)
                
            H = self.findHomography(-roll, pitch, 0., np.array([[0.], [dz - 13.5], [0.]])) #13.5-z
            warp = cv2.warpPerspective(frame, H, (960,720))
        
            rgbImage = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            self.changePixmap.emit(convertToQtFormat)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()

    def stop(self):
        self.ser.close()

class UI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.showMaximized()
        
        # Variables
        self.connStatus = 0

    def setImage(self, image):
        self.videostream.setPixmap(QPixmap.fromImage(image))

    def initUI(self):

        # Main Window
        palette = QPalette()
        self.setPalette(palette)
        self.setGeometry(100, 100, 1200, 960) # x-position, y-position, width, height
        self.setWindowTitle('Claw-Wheel Video Stabilization')

        # Port selection
        self.ComInput = QPlainTextEdit("COM", self)
        self.ComInput.setGeometry(QtCore.QRect(1000, 10, 120, 40))

        # Serial button
        ConnButton = QPushButton("CONNECT", self)
        ConnButton.setGeometry(QtCore.QRect(1000, 100, 120, 40))
        ConnButton.clicked.connect(self.ConnClicked)

        # Video stream window "label"
        self.videostream = QLabel(self)
        self.videostream.setGeometry(10, 10, 960, 720)

        # Warped image window
        self.warped = QLabel(self)
        self.warped.setGeometry(10, 800, 960, 120)

    def ConnClicked(self):
        if self.connStatus == 0:
            self.connStatus = 1

            # Video stream thread
            ComNumber = self.ComInput.toPlainText()
            self.video_thread = Capture(ComNumber) #Capture(self)
            self.video_thread.changePixmap.connect(self.setImage)
            self.video_thread.start()

        elif self.connStatus == 1:
            self.connStatus = 0
            self.video_thread.stop()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = UI()
    sys.exit(app.exec_())

"""
           task = SerialComm(ComNumber)
           QtCore.QThreadPool.globalInstance().start(task)

###################            
    class SerialComm(QtCore.QRunnable):
    def __init__(self, ComNumber):

        #super().__init__()
        QtCore.QRunnable.__init__(self)
        print(ComNumber)
        self.roll = 0
        self.pitch = 0

        self.ser = serial.Serial(ComNumber, 115200, timeout = 0.01)
        if(self.ser.isOpen() == True):
            self.ser.close()

        self.ser.open()
        print("Serial connected")
   
    def run(self):
        while True:
            self.ser.write(b'1')
            txt = self.ser.readline().decode('utf-8')
            self.roll = self.roll + 0.1
            if(txt):
                print(txt)

"""
