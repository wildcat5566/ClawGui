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
from PyQt5.QtGui import *

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
        
        self.relay = 0
        self.angvel = 0
        
        self.theta = 0
        self.roll = 0
        self.pitch = 0

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

    def encodeCovariance(self, angles):
        bytemsg = ''
        
        for angle in angles:
            b = 0
            while angle < 1:
                angle = angle * 10
                b = b + 1
            a = int(angle*10)
            bytemsg = bytemsg + str(a) + str(b)

        return bytemsg.encode('utf-8')

    def encodeComm(self, relay, angvel):
        bytemsg = str(relay)
        if angvel >= 100:
            bytemsg = bytemsg + str(angvel)
        elif angvel >= 10:
            bytemsg = bytemsg + '0' + str(angvel)
        else:
            bytemsg = bytemsg + '00' + str(angvel)

        return bytemsg.encode('utf-8')

    ### Control panel ###
    def RelayControl(self):
        if self.relay == 0:
            self.relay = 1
            print("relay on")
        elif self.relay == 1:
            self.relay = 0
            print("relay off")

    def VelocityControl(self, angvel):
        self.angvel = angvel

    ### Flow ###    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 960)
        cap.set(4, 720)
        
        Q_angle = 0.1
        Q_bias = 0.003
        R_measure = 0.02

        txt = self.ser.readline().decode('utf-8') # get rid of garbage
                    
        cov = self.encodeCovariance([Q_angle, Q_bias, R_measure])
        self.ser.write(cov) #103303202
        txt = self.ser.readline().decode('utf-8')
        print("Q_angle, Q_bias, R_measure:") # ensure covariances to be successfully set
        print(txt)
        while 1:
            ret, frame = cap.read()
            comm_msg = self.encodeComm(self.relay, self.angvel)
            print(comm_msg)
            self.ser.write(comm_msg)
                
            txt = self.ser.readline()
            if(txt):
                values = txt.decode('utf-8').split(',')
                print(values)
                self.theta = float(values[0])%180
                self.roll = float(values[1])
                self.pitch = float(values[2])
                dz = self.findZ(self.theta)
                
                H = self.findHomography(-self.roll, self.pitch, 0., np.array([[0.], [dz - 13.5], [0.]])) #13.5-z
                warp = cv2.warpPerspective(frame, H, (960,720))
        
                rgbImage = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                self.changePixmap.emit(convertToQtFormat)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()

class UI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.showMaximized()

    def setImage(self, image):
        self.videostream.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        ### Main Window ###
        palette = QPalette()
        self.setPalette(palette)
        self.setGeometry(100, 100, 1200, 960) # x-position, y-position, width, height
        self.setWindowTitle('Claw-Wheel Video Stabilization')

         #universal fonts
        font = QFont()
        font.setFamily("Calibri")
        font.setPointSize(16)

        # Video stream window "label"
        self.videostream = QLabel(self)
        self.videostream.setGeometry(10, 190, 960, 720)
        self.videostream.setStyleSheet("border: 1px solid black")

        ### Control panel ###
        # Port selection
        self.PortInput = QPlainTextEdit("COM28", self)
        self.PortInput.setGeometry(QtCore.QRect(10, 10, 120, 40))
        self.PortInput.setFont(font)

        # Connect button
        ConnButton = QPushButton("CONNECT", self)
        ConnButton.setGeometry(QtCore.QRect(150, 10, 120, 40))
        ConnButton.clicked.connect(self.ConnClicked)
        ConnButton.setFont(font)

        # Relay button
        RelayButton = QPushButton("RELAY", self)
        RelayButton.setGeometry(QtCore.QRect(10, 90, 120, 40))
        RelayButton.clicked.connect(self.RelayClicked)
        RelayButton.setFont(font)

        # Velocity label & slider
        self.VelLabel = QLabel(self)
        self.VelLabel.setGeometry(QtCore.QRect(150, 90, 200, 40))
        self.VelLabel.setText("Angular Velocity: 0")
        self.VelLabel.setFont(font)
        
        VelSlider = QSlider(QtCore.Qt.Horizontal, self)
        VelSlider.setGeometry(360, 80, 240, 60)
        VelSlider.setTickPosition(QSlider.TicksBothSides)
        VelSlider.setTickInterval(50)
        VelSlider.setMinimum(0)
        VelSlider.setMaximum(500)
        VelSlider.setSingleStep(10)
        VelSlider.setValue(0)
        VelSlider.valueChanged[int].connect(self.SetVelocity)

        # Cropping motion control
        # WASD
        # dist size <--> Zoom percentage?
        CropLabel = QLabel(self)
        CropLabel.setGeometry(QtCore.QRect(1000, 10, 300, 180))
        CropLabel.setText("Cropping motion control")
        CropLabel.setFont(font)
        CropLabel.setStyleSheet("border: 1px solid black")
        CropLabel.setAlignment(QtCore.Qt.AlignTop)

        UpButton = QPushButton("Up", self)
        UpButton.setGeometry(QtCore.QRect(1120, 50, 60, 60))
        UpButton.clicked.connect(self.UpClicked)
        UpButton.setFont(font)
        
        DownButton = QPushButton("Down", self)
        DownButton.setGeometry(QtCore.QRect(1120, 120, 60, 60))
        DownButton.clicked.connect(self.DownClicked)
        DownButton.setFont(font)
        
        LeftButton = QPushButton("Left", self)
        LeftButton.setGeometry(QtCore.QRect(1050, 85, 60, 60))
        LeftButton.clicked.connect(self.LeftClicked)
        LeftButton.setFont(font)
        
        RightButton = QPushButton("Right", self)
        RightButton.setGeometry(QtCore.QRect(1190, 85, 60, 60))
        RightButton.clicked.connect(self.RightClicked)
        RightButton.setFont(font)

        ### Kalman options ###
        # Q_angle, Q_bias, R_measure
        KalmanLabel = QLabel(self)
        KalmanLabel.setGeometry(QtCore.QRect(1000, 210, 300, 180))
        KalmanLabel.setText("Kalman Options \n Q_angle   Q_bias  R_measure")
        KalmanLabel.setFont(font)
        KalmanLabel.setStyleSheet("border: 1px solid black")
        KalmanLabel.setAlignment(QtCore.Qt.AlignTop)

        ### Spline options ###
        # Roll, pitch sampling ratio
        SplineLabel = QLabel(self)
        SplineLabel.setGeometry(QtCore.QRect(1000, 410, 300, 180))
        SplineLabel.setText("Spline Options")
        SplineLabel.setFont(font)
        SplineLabel.setStyleSheet("border: 1px solid black")
        SplineLabel.setAlignment(QtCore.Qt.AlignTop)

        ### Feature detect options ###
        # features count, features resolution, features distance, match distance
        FeatureLabel = QLabel(self)
        FeatureLabel.setGeometry(QtCore.QRect(1000, 610, 300, 300))
        FeatureLabel.setText("Feature Options")
        FeatureLabel.setFont(font)
        FeatureLabel.setStyleSheet("border: 1px solid black")
        FeatureLabel.setAlignment(QtCore.Qt.AlignTop)
        
    def ConnClicked(self):
        print("connect")
        # Video stream thread
        port = self.PortInput.toPlainText()
        self.video_thread = Capture(port)
        self.video_thread.changePixmap.connect(self.setImage)
        self.video_thread.start()

    def RelayClicked(self):
        self.video_thread.RelayControl()

    def SetVelocity(self, vel):
        self.video_thread.VelocityControl(vel)
        self.VelLabel.setText("Angular Velocity: " + str(vel))
        print(vel)

    def UpClicked(self):
        pass

    def DownClicked(self):
        pass

    def LeftClicked(self):
        pass

    def RightClicked(self):
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = UI()
    sys.exit(app.exec_())

