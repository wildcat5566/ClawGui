import sys
import time
import serial

import cv2
import numpy as np
import numpy.linalg as la
import cubicspline as cs
import ransac

from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class Capture(QtCore.QThread):
    changePixmap = QtCore.pyqtSignal(QImage)
    
    def __init__(self, ComNumber):
        QtCore.QThread.__init__(self)
        self.width = 960
        self.height = 720
        self.dist_width = 640
        self.dist_height = 480
        self.edge = 20
        
        self.f = None
        self.window = [self.height * 0.5, self.width * 0.5]
        self.lb = int(self.window[1]-0.5*self.dist_width)
        self.rb = int(self.window[1]+0.5*self.dist_width)
        self.tb = int(self.window[0]-0.5*self.dist_height)
        self.bb = int(self.window[0]+0.5*self.dist_height)

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
        self.r_tol = 0
        
        self.theta = 0
        self.roll = 0
        self.pitch = 0

        self.displayOpt = None

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

    ### Serial communication related ###
    def setArgs(self, Q_angle, Q_bias, R_measure, R_sample, P_sample, F_count, F_res, F_dist, M_dist, R_tol):
        self.q_angle = float(Q_angle)
        self.q_bias = float(Q_bias)
        self.r_measure = float(R_measure)

        self.r_sample = int(R_sample)
        self.p_sample = int(P_sample)

        self.f_count = int(F_count)
        self.f_res = float(F_res)
        self.f_dist = int(F_dist)
        self.m_dist = int(M_dist)
        self.r_tol = float(R_tol)
        
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

    def DisplayControl(self, opt):
        self.displayOpt = opt

    def WindowControl(self, horizontal, vertical):
        self.window[0] = self.window[0] + vertical*10
        self.window[1] = self.window[1] + horizontal*10

    ### Features match ###
    def plotFeatures(self, frame, new_f):
        cv2.line(frame, (self.lb, self.tb), (self.lb, self.bb), (255,255,0), 2) #bgr
        cv2.line(frame, (self.rb, self.tb), (self.rb, self.bb), (255,255,0), 2)
        cv2.line(frame, (self.lb, self.tb), (self.rb, self.tb), (255,255,0), 2)
        cv2.line(frame, (self.lb, self.bb), (self.rb, self.bb), (255,255,0), 2)

        cv2.circle(frame, (int(self.window[1]), int(self.window[0])), 3, (255,255,0), thickness=-1)

        for i in range(self.f.shape[0]):
            cv2.circle(frame, (int(self.f[i][0][0]), int(self.f[i][0][1])), 3, (0,0,255))

        for i in range(new_f.shape[0]):
            cv2.circle(frame, (int(new_f[i][0][0]), int(new_f[i][0][1])), 3, (0,0,255), thickness=-1)

    def matchFeaturesInit(self, frame):
        gray1 = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[self.edge:(self.height - self.edge), self.edge:(self.width - self.edge)]
        self.f = cv2.goodFeaturesToTrack(gray1, self.f_count, self.f_res, self.f_dist, useHarrisDetector=True)
        
    def matchFeatures(self, frame):
        t = time.time()
        gray2 = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[self.edge:(self.height - self.edge), self.edge:(self.width - self.edge)]
        new_f = cv2.goodFeaturesToTrack(gray2, self.f_count, self.f_res, self.f_dist, useHarrisDetector=True)

        #avoid boundaries
        for i in range(self.f.shape[0]):
            self.f[i][0][0] = self.f[i][0][0] + self.edge
            self.f[i][0][1] = self.f[i][0][1] + self.edge

        for i in range(new_f.shape[0]):
            new_f[i][0][0] = new_f[i][0][0] + self.edge
            new_f[i][0][1] = new_f[i][0][1] + self.edge
       
        matches = []
        for i in range(self.f.shape[0]):
            for j in range(i, new_f.shape[0]):
                x0 = self.f[i][0][0]
                y0 = self.f[i][0][1]
                x1 = new_f[j][0][0]
                y1 = new_f[j][0][1]
                dist2 = (x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0)
                if(dist2 < self.m_dist):
                    matches.append([x0, y0, x1, y1])

        print(len(matches))
        dispField = ransac.ransac_init(np.array(matches), self.r_tol)
        dispField = ransac.ransac(np.array(matches), dispField, 2, self.r_tol)
        dispField = ransac.ransac(np.array(matches), dispField, 3, self.r_tol)
    
        self.window[0] = self.window[0] + dispField[1]*1.0 #y
        self.window[1] = self.window[1] + dispField[0]*1.0

        self.lb = int(self.window[1]-0.5*self.dist_width)
        self.rb = int(self.window[1]+0.5*self.dist_width)
        self.tb = int(self.window[0]-0.5*self.dist_height)
        self.bb = int(self.window[0]+0.5*self.dist_height)

        # Check if exceeds boundary
        exc = 0
        if self.tb < 0:
            print('tb:' + str(self.tb))
            exc = 1
            self.window[0] = self.window[0] - self.tb
            self.bb = self.bb - self.tb
            self.tb = 0
            
        elif self.bb > self.height:
            exc = 1
            print('bb:' + str(self.bb))
            self.window[0] = self.window[0] - (self.bb - self.height)
            self.bb = self.height
            self.tb = self.bb - self.dist_height
            
        if self.lb < 0:
            exc = 1
            print('lb:' + str(self.lb))
            self.window[1] = self.window[1] - self.lb 
            self.rb = self.rb - self.lb
            self.lb = 0
            
        elif self.rb > self.width:
            exc = 1
            print('rb:' + str(self.rb))
            self.window[1] = self.window[1] - (self.rb - self.width)
            self.rb = self.width
            self.lb = self.rb - self.dist_width

        self.plotFeatures(frame, new_f)

        # Recover edge shift
        for i in range(new_f.shape[0]):
            new_f[i][0][0] = new_f[i][0][0] - self.edge
            new_f[i][0][1] = new_f[i][0][1] - self.edge

        elapsed = time.time() - t
        print(elapsed)
        ###Timing over###
        
        self.f = new_f

    ### Flow ###    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, self.width)
        cap.set(4, self.height)
        cap.set(6, 30)

        # Serial communication
        txt = self.ser.readline().decode('utf-8') # clear serial buffer
        cov = self.encodeCovariance([self.q_angle, self.q_bias, self.r_measure])
        self.ser.write(cov)
        txt = self.ser.readline().decode('utf-8')
        print("Q_angle, Q_bias, R_measure:") # ensure covariances to be successfully set
        print(txt)

        # Feature match initialization
        ret, frame = cap.read()
        self.matchFeaturesInit(frame)
        
        while 1:
            ret, frame = cap.read()
            comm_msg = self.encodeComm(self.relay, self.angvel)
            self.ser.write(comm_msg)
            txt = self.ser.readline()
            if(txt):
                # Retrieve arduino data
                values = txt.decode('utf-8').split(',')
                #print(self.displayOpt)
                self.theta = float(values[0])%180
                self.roll = float(values[1])
                self.pitch = float(values[2])
                dz = self.findZ(self.theta)

                # Warp first
                H = self.findHomography(-self.roll, self.pitch, 0., np.array([[0.], [dz - 13.5], [0.]])) #13.5-z
                warp = cv2.warpPerspective(frame, H, (self.width, self.height))

                # Find features and crop
                self.matchFeatures(warp)

                # Final output
                crop = warp[self.tb:self.bb, self.lb:self.rb, :]
                if self.displayOpt == True:
                    rgbImage = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    rgbImage = cv2.resize(rgbImage, (960, 720))
                    
                else:
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
        self.showContent = False

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
        RelayButton.setGeometry(QtCore.QRect(290, 10, 120, 40))
        RelayButton.clicked.connect(self.RelayClicked)
        RelayButton.setFont(font)

        # Velocity label & slider
        self.VelLabel = QLabel(self)
        self.VelLabel.setGeometry(QtCore.QRect(10, 70, 200, 40))
        self.VelLabel.setText("Angular Velocity: 0")
        self.VelLabel.setFont(font)
        
        VelSlider = QSlider(QtCore.Qt.Horizontal, self)
        VelSlider.setGeometry(250, 60, 240, 60)
        VelSlider.setTickPosition(QSlider.TicksBothSides)
        VelSlider.setTickInterval(50)
        VelSlider.setMinimum(0)
        VelSlider.setMaximum(500)
        VelSlider.setSingleStep(10)
        VelSlider.setValue(0)
        VelSlider.valueChanged[int].connect(self.SetVelocity)

        # Show options
       
        self.ShowCrop = QRadioButton('Show cropped', self)
        self.ShowCrop.setGeometry(QtCore.QRect(250, 120, 200, 40))
        self.ShowCrop.setFont(font)
        self.ShowCrop.toggle()  
        self.ShowCrop.toggled.connect(self.CropChecked)
        
        self.ShowOrig = QRadioButton('Show original', self)
        self.ShowOrig.setGeometry(QtCore.QRect(10, 120, 200, 40))
        self.ShowOrig.setFont(font)
        self.ShowOrig.toggle()  
        self.ShowOrig.toggled.connect(self.OrigChecked)
 

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
        FeatureLabel.setText("Feature Options \n Feature count \n \n Feature res. \n \n Feature dist. \n \n min. Match dist. \n \n Ransac tol.")
        FeatureLabel.setFont(font)
        FeatureLabel.setStyleSheet("border: 1px solid black")
        FeatureLabel.setAlignment(QtCore.Qt.AlignTop)
        self.FcountInput = QPlainTextEdit("50", self)
        self.FcountInput.setGeometry(QtCore.QRect(1160, 640, 120, 40))
        self.FcountInput.setFont(font)
        self.FresInput = QPlainTextEdit("0.01", self)
        self.FresInput.setGeometry(QtCore.QRect(1160, 690, 120, 40))
        self.FresInput.setFont(font)
        self.FdistInput = QPlainTextEdit("30", self)
        self.FdistInput.setGeometry(QtCore.QRect(1160, 740, 120, 40))
        self.FdistInput.setFont(font)
        self.MdistInput = QPlainTextEdit("2000", self)
        self.MdistInput.setGeometry(QtCore.QRect(1160, 790, 120, 40))
        self.MdistInput.setFont(font)
        self.RtolInput = QPlainTextEdit("1.25", self)
        self.RtolInput.setGeometry(QtCore.QRect(1160, 840, 120, 40))
        self.RtolInput.setFont(font)

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
        R_tol = self.RtolInput.toPlainText()
        
        self.video_thread.setArgs(Q_angle, Q_bias, R_measure, R_sample, P_sample, F_count, F_res, F_dist, M_dist, R_tol)
        self.video_thread.DisplayControl(False)

    def RelayClicked(self):
        self.video_thread.RelayControl()

    def SetVelocity(self, vel):
        self.video_thread.VelocityControl(vel)
        self.VelLabel.setText("Angular Velocity: " + str(vel))
        print(vel)

    def CropChecked(self, value):
        if self.ShowCrop.isChecked():
            self.ShowOrig.setChecked(False)
            
        else:
            self.ShowOrig.setChecked(True)
            
            
    def OrigChecked(self, value):
        if self.ShowOrig.isChecked():
            self.ShowCrop.setChecked(False)
            self.video_thread.DisplayControl(False)
        else:
            self.ShowCrop.setChecked(True)
            self.video_thread.DisplayControl(True)

    def UpClicked(self):
        self.video_thread.WindowControl(0, -1)

    def DownClicked(self):
        self.video_thread.WindowControl(0, 1)

    def LeftClicked(self):
        self.video_thread.WindowControl(-1, 0)

    def RightClicked(self):
        self.video_thread.WindowControl(1, 0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = UI()
    sys.exit(app.exec_())

