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

        self.q_angle = 0
        self.q_bias = 0
        self.r_measure = 0

        self.r_sample = 0
        self.p_sample = 0

        self.f_count = 0
        self.f_res = 0
        self.f_dist = 0
        self.m_dist = 0
        
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
        bytemsg = ''
        if angvel >= 100:
            bytemsg = bytemsg + str(angvel) + str(relay)
        elif angvel >= 10:
            bytemsg = bytemsg + '0' + str(angvel) + str(relay)
        else:
            bytemsg = bytemsg + '00' + str(angvel) + str(relay)

        return bytemsg.encode('utf-8')

    def setArgs(self, Q_angle, Q_bias, R_measure, R_sample, P_sample, F_count, F_res, F_dist, M_dist):
        self.q_angle = float(Q_angle)
        self.q_bias = float(Q_bias)
        self.r_measure = float(R_measure)

        self.r_sample = int(R_sample)
        self.p_sample = int(P_sample)

        self.f_count = int(F_count)
        self.f_res = float(F_res)
        self.f_dist = int(F_dist)
        self.m_dist = int(M_dist)

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

        txt = self.ser.readline().decode('utf-8') # get rid of garbage
                    
        cov = self.encodeCovariance([self.q_angle, self.q_bias, self.r_measure])
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
        KalmanLabel.setText("Kalman Options \n Q_angle \n \n Q_bias \n \n R_measure")
        KalmanLabel.setFont(font)
        KalmanLabel.setStyleSheet("border: 1px solid black")
        KalmanLabel.setAlignment(QtCore.Qt.AlignTop)
        self.QangleInput = QPlainTextEdit("0.1", self)
        self.QangleInput.setGeometry(QtCore.QRect(1160, 240, 120, 40))
        self.QangleInput.setFont(font)
        self.QbiasInput = QPlainTextEdit("0.003", self)
        self.QbiasInput.setGeometry(QtCore.QRect(1160, 290, 120, 40))
        self.QbiasInput.setFont(font)
        self.RmeasureInput = QPlainTextEdit("0.02", self)
        self.RmeasureInput.setGeometry(QtCore.QRect(1160, 340, 120, 40))
        self.RmeasureInput.setFont(font)

        ### Spline options ###
        # Roll, pitch sampling ratio
        SplineLabel = QLabel(self)
        SplineLabel.setGeometry(QtCore.QRect(1000, 410, 300, 180))
        SplineLabel.setText("Spline Options (Sampling Ratio) \n Roll \n \n Pitch")
        SplineLabel.setFont(font)
        SplineLabel.setStyleSheet("border: 1px solid black")
        SplineLabel.setAlignment(QtCore.Qt.AlignTop)
        self.RSampleInput = QPlainTextEdit("10", self)
        self.RSampleInput.setGeometry(QtCore.QRect(1160, 440, 120, 40))
        self.RSampleInput.setFont(font)
        self.PSampleInput = QPlainTextEdit("10", self)
        self.PSampleInput.setGeometry(QtCore.QRect(1160, 490, 120, 40))
        self.PSampleInput.setFont(font)

        ### Feature detect options ###
        # features count, features resolution, features distance, match distance
        FeatureLabel = QLabel(self)
        FeatureLabel.setGeometry(QtCore.QRect(1000, 610, 300, 300))
        FeatureLabel.setText("Feature Options \n Feature count \n \n Feature res. \n \n Feature dist. \n \n min. Match dist.")
        FeatureLabel.setFont(font)
        FeatureLabel.setStyleSheet("border: 1px solid black")
        FeatureLabel.setAlignment(QtCore.Qt.AlignTop)
        self.FcountInput = QPlainTextEdit("50", self)
        self.FcountInput.setGeometry(QtCore.QRect(1160, 640, 120, 40))
        self.FcountInput.setFont(font)
        self.FresInput = QPlainTextEdit("0.1", self)
        self.FresInput.setGeometry(QtCore.QRect(1160, 690, 120, 40))
        self.FresInput.setFont(font)
        self.FdistInput = QPlainTextEdit("30", self)
        self.FdistInput.setGeometry(QtCore.QRect(1160, 740, 120, 40))
        self.FdistInput.setFont(font)
        self.MdistInput = QPlainTextEdit("1200", self)
        self.MdistInput.setGeometry(QtCore.QRect(1160, 790, 120, 40))
        self.MdistInput.setFont(font)

    def ConnClicked(self):
        print("connect")
        # Video stream thread
        port = self.PortInput.toPlainText()
        self.video_thread = Capture(port)
        self.video_thread.changePixmap.connect(self.setImage)
        self.video_thread.start()

        # Send arguments
        Q_angle = self.QangleInput.toPlainText()
        Q_bias = self.QbiasInput.toPlainText()
        R_measure = self.RmeasureInput.toPlainText()
        R_sample = self.RSampleInput.toPlainText()
        P_sample = self.PSampleInput.toPlainText()
        F_count = self.FcountInput.toPlainText()
        F_res = self.FresInput.toPlainText()
        F_dist = self.FdistInput.toPlainText()
        M_dist = self.MdistInput.toPlainText()
        self.video_thread.setArgs(Q_angle, Q_bias, R_measure, R_sample, P_sample, F_count, F_res, F_dist, M_dist)

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

