import sys
import time
import serial

import cv2
import numpy as np
import numpy.linalg as la
import spline
import ransac

from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class Capture(QtCore.QThread):
    changePixmap = QtCore.pyqtSignal(QImage)
    
    def __init__(self, ComNumber):
        QtCore.QThread.__init__(self)
        self.width = 800
        self.height = 600
        self.dist_width = 480
        self.dist_height = 360
        self.edge = 50
        
        self.f = None
        self.window = [self.height * 0.5, self.width * 0.5]
        self.vb = [int(self.window[1]-0.5*self.dist_width), int(self.window[1]+0.5*self.dist_width)]
        self.hb = [int(self.window[0]-0.5*self.dist_height), int(self.window[0]+0.5*self.dist_height)]
        self.dispField = [0, 0]

        self.K = np.matrix([[756.78768558,   0.        , 629.89805344],
                            [  0.        , 756.86336981, 345.49169401],
                            [  0.        ,   0.        ,   1.        ]])
        self.d = 50
        
        self.ser = serial.Serial(ComNumber, 115200, timeout = 0.01)
        if(self.ser.isOpen() == True):
            self.ser.close()
        self.ser.open()
        
        self.relay = 0
        self.angvel = 0

        self.q_angle = 0
        self.q_bias = 0
        self.r_measure = 0

        self.s_sample = 0
        self.s_alpha = 0
        self.bench = None

        self.f_count = 0
        self.f_res = 0
        self.f_dist = 0
        self.m_dist = 0
        self.r_tol = 0
        self.r_ite = 0
        
        self.theta = None
        self.roll = None
        self.pitch = None
        self.dx = None
        self.dy = None
        
        self.t = 0
        self.r = 0
        self.p = 0

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
        
    
        H = self.K*(R + t*np.transpose(n)/self.d)*la.inv(self.K)
        return(H)

    ### Serial communication related ###
    def setArgs(self, Q_angle, Q_bias, R_measure, S_sample, S_alpha, F_count, F_res, F_dist, M_dist, R_tol, R_ite, angvel):
        self.q_angle = float(Q_angle)
        self.q_bias = float(Q_bias)
        self.r_measure = float(R_measure)
        self.s_sample = int(S_sample)
        self.s_alpha = float(S_alpha)
        self.f_count = int(F_count)
        self.f_res = float(F_res)
        self.f_dist = int(F_dist)
        self.m_dist = int(M_dist)
        self.r_tol = float(R_tol)
        self.r_ite = int(R_ite)
        self.m_count = 0
        self.angvel = angvel

        self.theta = np.zeros((self.s_sample * 3 + 1))
        self.roll = np.zeros((self.s_sample * 3 + 1))
        self.pitch = np.zeros((self.s_sample * 3 + 1))
        
        self.dx = np.zeros((self.s_sample))
        self.dy = np.zeros((self.s_sample))

        self.bench = np.zeros((self.s_sample * 2, self.height, self.width, 3),dtype=np.uint8)

    def encodeStartMsg(self, angles, angvel):
        bytemsg = ''
        
        for angle in angles:
            b = 0
            while angle < 1:
                angle = angle * 10
                b = b + 1
            a = int(angle*10)
            bytemsg = bytemsg + str(a) + str(b)

        if angvel >= 100:
            bytemsg = bytemsg + str(angvel) 
        elif angvel >= 10:
            bytemsg = bytemsg + '0' + str(angvel)
        else:
            bytemsg = bytemsg + '00' + str(angvel)

        return bytemsg.encode('utf-8')

    def encodeMsg(self, relay):
        if(relay==1):
            bytemsg = '1'
        else:
            bytemsg = '12'
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

    ### Drawing Functions ###
    def plotFeatures(self, frame, new_f):
        bound_col = []
        for i in range(4):
            bound_col.append((255,255,0)) #bgr

        cv2.line(frame, (self.vb[0],   self.hb[0]),   (self.vb[1]-1, self.hb[0]  ), bound_col[0], 2)
        cv2.line(frame, (self.vb[0],   self.hb[1]-1), (self.vb[1]-1, self.hb[1]-1), bound_col[1], 2)
        cv2.line(frame, (self.vb[0],   self.hb[0]),   (self.vb[0],   self.hb[1]  ), bound_col[2], 2) 
        cv2.line(frame, (self.vb[1]-1, self.hb[0]),   (self.vb[1]-1, self.hb[1]-1), bound_col[3], 2)

        cv2.circle(frame, (int(self.window[1]), int(self.window[0])), 3, (255,255,0), thickness=-1)

        for i in range(self.f.shape[0]):
            cv2.circle(frame, (int(self.f[i][0][0]), int(self.f[i][0][1])), 3, (0,0,255))

        for i in range(new_f.shape[0]):
            cv2.circle(frame, (int(new_f[i][0][0]), int(new_f[i][0][1])), 3, (0,0,255), thickness=-1)

    def plotBound(self, frame):        
        cv2.line(frame, (self.vb[0],   self.hb[0]),   (self.vb[1]-1, self.hb[0]  ), (0,0,0), 1)
        cv2.line(frame, (self.vb[0],   self.hb[1]-1), (self.vb[1]-1, self.hb[1]-1), (0,0,0), 1)
        cv2.line(frame, (self.vb[0],   self.hb[0]),   (self.vb[0],   self.hb[1]  ), (0,0,0), 1) 
        cv2.line(frame, (self.vb[1]-1, self.hb[0]),   (self.vb[1]-1, self.hb[1]-1), (0,0,0), 1)
        cv2.circle(frame, (int(self.window[1]), int(self.window[0])), 1, (0,0,0), thickness=-1)

    ### Feature matching ###
    def matchFeaturesInit(self, frame):
        gray = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[self.edge:(self.height - self.edge), self.edge:(self.width - self.edge)]
        self.f = cv2.goodFeaturesToTrack(gray, self.f_count, self.f_res, self.f_dist, useHarrisDetector=True)
        
    def matchFeatures(self, frame): # Returns displacement field
        gray = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[self.edge:(self.height - self.edge), self.edge:(self.width - self.edge)]
        new_f = cv2.goodFeaturesToTrack(gray, self.f_count, self.f_res, self.f_dist, useHarrisDetector=True)
        # Shift to avoid boundaries
        for i in range(self.f.shape[0]):
            self.f[i][0][0] = self.f[i][0][0] + self.edge
            self.f[i][0][1] = self.f[i][0][1] + self.edge

        for i in range(new_f.shape[0]):
            new_f[i][0][0] = new_f[i][0][0] + self.edge
            new_f[i][0][1] = new_f[i][0][1] + self.edge

        #(1) Match between consequential frames
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

        #(2) Find displacement field with Ransac
        self.m_count = len(matches)
        self.dispField = ransac.ransac_init(np.array(matches), self.r_tol)
        for i in range(1, self.r_ite + 1):
            self.dispField = ransac.ransac(np.array(matches), self.dispField, i, self.r_tol)
        
        #self.plotFeatures(frame, new_f)
        # Recover boundary shift
        for i in range(new_f.shape[0]):
            new_f[i][0][0] = new_f[i][0][0] - self.edge
            new_f[i][0][1] = new_f[i][0][1] - self.edge
        
        self.f = new_f

    ### Flow ###    
    def run(self):
        print("### main flow start ###")
        cap = cv2.VideoCapture(0)
        cap.set(3, self.width)
        cap.set(4, self.height)
        cap.set(6, 30)

        # Initialize serial communication
        txt = self.ser.readline().decode('utf-8') # clear serial buffer
        start_msg = self.encodeStartMsg([self.q_angle, self.q_bias, self.r_measure], self.angvel)
        self.ser.write(start_msg)
        txt = self.ser.readline().decode('utf-8')
        print("Q_angle, Q_bias, R_measure:") # ensure covariances to be successfully set
        print(txt)

        # Populate data: IMU
        i = 0
        while i < (self.s_sample*2):
            msg = self.encodeMsg(self.relay)
            self.ser.write(msg)
            txt = self.ser.readline() 
            if(txt):
                
                values = txt.decode('utf-8').split(',')
                self.theta[i] = float(values[0])%180
                self.roll[i] = float(values[1])
                self.pitch[i] = float(values[2])
                ret, frame = cap.read()
                
                if i==0:
                    self.matchFeaturesInit(frame)
                    
                if i >= self.s_sample:
                    self.bench[i - self.s_sample] = frame
                    self.matchFeatures(frame)

                i = i + 1 # Proceed if txt successfully received; else retry
        
        t = time.time()

        # Initialize spline objects & targets
        pitch_spline = spline.Spline(self.s_sample, self.s_alpha)
        roll_spline = spline.Spline(self.s_sample, self.s_alpha)
        
        pitch_tar = pitch_spline.findSpline(self.pitch[0], self.pitch[0], self.pitch[self.s_sample], self.pitch[2*self.s_sample - 1])
        roll_tar = roll_spline.findSpline(self.roll[0], self.roll[0], self.roll[self.s_sample], self.roll[2*self.s_sample - 1])

        # Start main flow
        nth = 0
        fcount = 0

        while 1:
            msg = self.encodeMsg(self.relay)
            self.ser.write(b'1')
            txt = self.ser.readline()
            if(txt):
                #(1) Collect data & Store into waiting line
                values = txt.decode('utf-8').split(',')
                self.t = float(values[0])%180
                self.r = float(values[1])
                self.p = float(values[2])

                self.theta[2*self.s_sample + nth] = float(values[0])%180
                self.roll[2*self.s_sample + nth] = float(values[1])
                self.pitch[2*self.s_sample + nth] = float(values[2])
                ret, frame = cap.read()
                self.bench[self.s_sample + nth] = frame
                ### Save screenshot ###
                fname = '../data/0629/unstab/' + str(fcount) + '.jpg'
                cv2.imwrite(fname, self.bench[nth])

                #(2) Warp based on spline targets
                roll_bias = - self.roll[self.s_sample + nth] + roll_tar[nth]
                pitch_bias = self.pitch[self.s_sample + nth] - pitch_tar[nth]

                dz = self.findZ(self.theta[self.s_sample + nth])
                H = self.findHomography(roll_bias, pitch_bias, 0., np.array([[0.], [13.5 - dz], [0.]])) #13.5-z
                warped = cv2.warpPerspective(self.bench[nth], H, (self.width, self.height))

                #(3) Find & track features
                self.matchFeatures(warped)
                self.window[0] = self.window[0] + self.dispField[1]*1.0 #y
                self.window[1] = self.window[1] + self.dispField[0]*1.0

                #(4) cover inherent motion in pixelwise
                # pitch
                #self.window[0] = self.window[0] - self.d * self.sin(pitch_bias)
                #self.hb = [int(self.window[0]-0.5*self.dist_height), int(self.window[0]+0.5*self.dist_height)]

                #(5) Check if exceeds boundary
                exc = [0,0,0,0]
                if self.window[0] - 0.5*self.dist_height <= 0: #up
                    self.window[0] = 0.5*self.dist_height
                    exc[0] = 1

                elif self.window[0]+0.5*self.dist_height >= self.height: #down
                    self.window[0] = self.height - (0.5*self.dist_height)
                    exc[1] = 1

                if (self.window[1]-0.5*self.dist_width) <= 0: #left
                    self.window[1] = 0.5*self.dist_width
                    exc[2] = 1
            
                elif (self.window[1]+0.5*self.dist_width) >= self.width: #right
                    self.window[1] = self.width - (0.5*self.dist_width)
                    exc[3] = 1

                self.vb = [int(self.window[1]-0.5*self.dist_width), int(self.window[1]-0.5*self.dist_width) + self.dist_width]
                self.hb = [int(self.window[0]-0.5*self.dist_height), int(self.window[0]-0.5*self.dist_height) + self.dist_height]

                #(6) Final crop
                crop = warped[self.hb[0]:(self.hb[0]+self.dist_height),
                              self.vb[0]:(self.vb[0]+self.dist_width), :]
                ### Save screenshot ###
                fname = '../data/0629/stab/' + str(fcount) + '.jpg'
                cv2.imwrite(fname, crop)

                #(7) Final display
                self.plotBound(warped)
                if self.displayOpt == False:
                    dispImage = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                else:
                    dispImage = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    
                dispImage = cv2.resize(dispImage, (960,720))
                convertToQtFormat = QImage(dispImage.data, dispImage.shape[1], dispImage.shape[0], QImage.Format_RGB888) #QImage.Format_Indexed8
                self.changePixmap.emit(convertToQtFormat)
                
                #(8) Update & push bench frames & sensor data
                if nth == self.s_sample - 1:
                    # Spline nodes
                    pitch_tar = pitch_spline.findSpline(self.pitch[0],               self.pitch[self.s_sample],
                                                        self.pitch[self.s_sample*2], self.pitch[self.s_sample*3 - 1])
                    roll_tar = roll_spline.findSpline(self.roll[0],               self.roll[self.s_sample],
                                                      self.roll[self.s_sample*2], self.roll[self.s_sample*3 - 1])

                    for i in range(self.s_sample*2):
                        self.theta[i] = self.theta[i+self.s_sample]
                        self.roll[i] = self.roll[i+self.s_sample]
                        self.pitch[i] = self.pitch[i+self.s_sample]

                    for i in range(self.s_sample):
                        self.bench[i] = self.bench[i+self.s_sample]

                    nth = 0
                else:
                    nth = nth + 1

                fcount = fcount + 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()

class UI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.showMaximized()
        self.angvel = 200
        
    def setImage(self, image):
        self.videostream.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        ### Main Window ###
        palette = QPalette()
        self.setPalette(palette)
        self.setGeometry(100, 100, 1200, 960) # x-position, y-position, width, height
        self.setWindowTitle('Claw-Wheel Video Stabilization')

        #universal font
        font = QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        self.setFont(font)

        # Video stream window "label"
        self.videostream = QLabel(self)
        self.videostream.setGeometry(10, 190, 960, 720)
        self.videostream.setStyleSheet("border: 1px solid black")

        ### Control panel ###
        # Port selection
        self.PortInput = QPlainTextEdit("COM4", self)
        self.PortInput.setGeometry(QtCore.QRect(10, 10, 120, 40))

        # Connect button
        ConnButton = QPushButton("CONNECT", self)
        ConnButton.setGeometry(QtCore.QRect(150, 10, 120, 40))
        ConnButton.clicked.connect(self.ConnClicked)

        # Relay button
        RelayButton = QPushButton("RELAY", self)
        RelayButton.setGeometry(QtCore.QRect(290, 10, 120, 40))
        RelayButton.clicked.connect(self.RelayClicked)

        # Velocity label & slider
        self.VelLabel = QLabel(self)
        self.VelLabel.setGeometry(QtCore.QRect(10, 70, 200, 40))
        self.VelLabel.setText("Angular Velocity: 200")
        
        VelSlider = QSlider(QtCore.Qt.Horizontal, self)
        VelSlider.setGeometry(250, 60, 240, 60)
        VelSlider.setTickPosition(QSlider.TicksBothSides)
        VelSlider.setTickInterval(50)
        VelSlider.setMinimum(0)
        VelSlider.setMaximum(400)
        VelSlider.setSingleStep(10)
        VelSlider.setValue(200)
        VelSlider.valueChanged[int].connect(self.SetVelocity)

        # Show options
        self.ShowCrop = QRadioButton('Show cropped', self)
        self.ShowCrop.setGeometry(QtCore.QRect(250, 120, 200, 40))
        self.ShowCrop.toggle()  
        self.ShowCrop.toggled.connect(self.CropChecked)
        
        self.ShowOrig = QRadioButton('Show original', self)
        self.ShowOrig.setGeometry(QtCore.QRect(10, 120, 200, 40))
        self.ShowOrig.toggle()  
        self.ShowOrig.toggled.connect(self.OrigChecked)

        # Cropping motion control
        # WASD
        # dist size <--> Zoom percentage?
        CropLabel = QLabel(self)
        CropLabel.setGeometry(QtCore.QRect(1000, 10, 300, 180))
        CropLabel.setText("Cropping motion control")
        CropLabel.setStyleSheet("border: 1px solid black")
        CropLabel.setAlignment(QtCore.Qt.AlignTop)

        UpButton = QPushButton("Up", self)
        UpButton.setGeometry(QtCore.QRect(1120, 50, 60, 60))
        UpButton.clicked.connect(self.UpClicked)
        
        DownButton = QPushButton("Down", self)
        DownButton.setGeometry(QtCore.QRect(1120, 120, 60, 60))
        DownButton.clicked.connect(self.DownClicked)
        
        LeftButton = QPushButton("Left", self)
        LeftButton.setGeometry(QtCore.QRect(1050, 85, 60, 60))
        LeftButton.clicked.connect(self.LeftClicked)
        
        RightButton = QPushButton("Right", self)
        RightButton.setGeometry(QtCore.QRect(1190, 85, 60, 60))
        RightButton.clicked.connect(self.RightClicked)

        ### Kalman options ###
        # Q_angle, Q_bias, R_measure
        KalmanLabel = QLabel(self)
        KalmanLabel.setGeometry(QtCore.QRect(1000, 210, 300, 180))
        KalmanLabel.setText("Kalman Options \n Q_angle \n \n Q_bias \n \n R_measure")
        KalmanLabel.setStyleSheet("border: 1px solid black")
        KalmanLabel.setAlignment(QtCore.Qt.AlignTop)
        self.QangleInput = QPlainTextEdit("0.01", self) #0.1
        self.QangleInput.setGeometry(QtCore.QRect(1160, 240, 120, 40))
        self.QbiasInput = QPlainTextEdit("0.003", self)
        self.QbiasInput.setGeometry(QtCore.QRect(1160, 290, 120, 40))
        self.RmeasureInput = QPlainTextEdit("0.005", self) #0.02
        self.RmeasureInput.setGeometry(QtCore.QRect(1160, 340, 120, 40))

        ### Spline options ###
        # Roll, pitch sampling ratio
        SplineLabel = QLabel(self)
        SplineLabel.setGeometry(QtCore.QRect(1000, 410, 300, 130))
        SplineLabel.setText("Spline Options \n Sampling ratio \n \n alpha")
        SplineLabel.setStyleSheet("border: 1px solid black")
        SplineLabel.setAlignment(QtCore.Qt.AlignTop)
        self.SplineSampleInput = QPlainTextEdit("10", self)
        self.SplineSampleInput.setGeometry(QtCore.QRect(1160, 440, 120, 40))
        self.SplineAlphaInput = QPlainTextEdit("0", self)
        self.SplineAlphaInput.setGeometry(QtCore.QRect(1160, 490, 120, 40))

        ### Feature detect options ###
        # features count, features resolution, features distance, match distance
        FeatureLabel = QLabel(self)
        FeatureLabel.setGeometry(QtCore.QRect(1000, 560, 300, 350))
        FeatureLabel.setText("Feature Options \n Feature count \n \n Feature res. \n \n Feature dist. \n \n min. Match dist. \n \n Ransac tol. \n \n Ransac ite.'s")
        FeatureLabel.setStyleSheet("border: 1px solid black")
        FeatureLabel.setAlignment(QtCore.Qt.AlignTop)
        self.FcountInput = QPlainTextEdit("100", self) #50
        self.FcountInput.setGeometry(QtCore.QRect(1160, 590, 120, 40))
        self.FresInput = QPlainTextEdit("0.001", self)
        self.FresInput.setGeometry(QtCore.QRect(1160, 640, 120, 40))
        self.FdistInput = QPlainTextEdit("30", self)
        self.FdistInput.setGeometry(QtCore.QRect(1160, 690, 120, 40))
        self.MdistInput = QPlainTextEdit("3000", self)
        self.MdistInput.setGeometry(QtCore.QRect(1160, 740, 120, 40))
        self.RtolInput = QPlainTextEdit("1.1", self)
        self.RtolInput.setGeometry(QtCore.QRect(1160, 790, 120, 40))
        self.RiteInput = QPlainTextEdit("3", self)
        self.RiteInput.setGeometry(QtCore.QRect(1160, 840, 120, 40))

        ### State labels ###
        StateLabel = QLabel(self)
        StateLabel.setGeometry(QtCore.QRect(580, 70, 400, 40))
        StateLabel.setText("Theta          Roll              Pitch           Matches")
        self.ThetaLabel = QLabel(self)
        self.ThetaLabel.setGeometry(QtCore.QRect(580, 120, 80, 40))
        self.ThetaLabel.setText("0")
        self.RollLabel = QLabel(self)
        self.RollLabel.setGeometry(QtCore.QRect(680, 120, 80, 40))
        self.RollLabel.setText("0")
        self.PitchLabel = QLabel(self)
        self.PitchLabel.setGeometry(QtCore.QRect(780, 120, 80, 40))
        self.PitchLabel.setText("0")
        self.MatchLabel = QLabel(self)
        self.MatchLabel.setGeometry(QtCore.QRect(880, 120, 80, 40))
        self.MatchLabel.setText("0")

    def Update(self):
        self.ThetaLabel.setText(str(self.video_thread.t))
        self.RollLabel.setText(str(self.video_thread.r))
        self.PitchLabel.setText(str(self.video_thread.p))
        self.MatchLabel.setText(str(self.video_thread.m_count))

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
        S_sample = self.SplineSampleInput.toPlainText()
        S_alpha = self.SplineAlphaInput.toPlainText()
        F_count = self.FcountInput.toPlainText()
        F_res = self.FresInput.toPlainText()
        F_dist = self.FdistInput.toPlainText()
        M_dist = self.MdistInput.toPlainText()
        R_tol = self.RtolInput.toPlainText()
        R_ite = self.RiteInput.toPlainText()
        
        self.video_thread.setArgs(Q_angle, Q_bias, R_measure, S_sample, S_alpha,
                                  F_count, F_res, F_dist, M_dist, R_tol, R_ite, self.angvel)
        self.video_thread.DisplayControl(False)

        self.SysSerialTimer = QtCore.QTimer()
        self.SysSerialTimer.timeout.connect(self.Update)
        self.SysSerialTimer.start(50)
        
    def RelayClicked(self):
        self.video_thread.RelayControl()

    def SetVelocity(self, vel):
        self.VelLabel.setText("Angular Velocity: " + str(vel))
        self.angvel = vel
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

