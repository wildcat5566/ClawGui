import cv2
import numpy as np
import numpy.linalg as la
import spline
import ransac

class Video():
    def __init__(self):
        self.width = 800
        self.height = 600
        self.dist_width = 480
        self.dist_height = 360
        self.edge = 0
        
        self.f = None
        self.window = [self.height * 0.5 - 80, self.width * 0.5]
        self.vb = [int(self.window[1]-0.5*self.dist_width), int(self.window[1]+0.5*self.dist_width)]
        self.hb = [int(self.window[0]-0.5*self.dist_height), int(self.window[0]+0.5*self.dist_height)]
        self.dispField = None

        self.f_count = 100
        self.f_res = 0.001
        self.f_dist = 50
        self.m_dist = 3000#3000
        self.r_tol = 1.5
        self.r_ite = 10
        
    def matchFeaturesInit(self, frame):
        print('0')
        gray = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[self.edge:(self.height - self.edge), self.edge:(self.width - self.edge)]
        self.f = cv2.goodFeaturesToTrack(gray, self.f_count, self.f_res, self.f_dist, useHarrisDetector=False)
        
        bound_col = []
        for i in range(4):
            bound_col.append((255,255,0)) #bgr
        cv2.line(frame, (self.vb[0],   self.hb[0]),   (self.vb[1]-1, self.hb[0]  ), bound_col[0], 2)
        cv2.line(frame, (self.vb[0],   self.hb[1]-1), (self.vb[1]-1, self.hb[1]-1), bound_col[1], 2)
        cv2.line(frame, (self.vb[0],   self.hb[0]),   (self.vb[0],   self.hb[1]  ), bound_col[2], 2) 
        cv2.line(frame, (self.vb[1]-1, self.hb[0]),   (self.vb[1]-1, self.hb[1]-1), bound_col[3], 2)
        cv2.circle(frame, (int(self.window[1]), int(self.window[0])), 3, (255,255,0), thickness=-1)

        cv2.imwrite('../data/0701/2/stab/0.jpg',frame)
        crop = frame[self.hb[0]:self.hb[1], self.vb[0]:self.vb[1], :]
        cv2.imwrite('../data/0701/2/crop/0.jpg', crop)
        
    def matchFeatures(self, frame, fcount): # Compare to previous features
        print(fcount)
        blur = cv2.GaussianBlur(frame, (3,3), 0);
        frame = cv2.addWeighted(frame, 2, blur, -1, 0);
        
        
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
            for j in range(new_f.shape[0]):
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

        #(3) Move window
        self.window[0] = self.window[0] + self.dispField[1]*1.0 #1:y
        self.window[1] = self.window[1] + self.dispField[0]*1.0

        #(4) Check if exceeds boundary
        if self.window[0] - 0.5*self.dist_height <= 0: #up
            self.window[0] = 0.5*self.dist_height

        elif self.window[0]+0.5*self.dist_height >= self.height: #down
            self.window[0] = self.height - (0.5*self.dist_height)

        if self.window[1]-0.5*self.dist_width <= 0: #left
            self.window[1] = 0.5*self.dist_width

        elif (self.window[1]+0.5*self.dist_width) >= self.width: #right
            self.window[1] = self.width - (0.5*self.dist_width)

        self.vb = [int(self.window[1]-0.5*self.dist_width), int(self.window[1]-0.5*self.dist_width) + self.dist_width]
        self.hb = [int(self.window[0]-0.5*self.dist_height), int(self.window[0]-0.5*self.dist_height) + self.dist_height]

        crop = frame[self.hb[0]:self.hb[1], self.vb[0]:self.vb[1], :]
        cv2.imwrite('../data/0701/2/crop/'+str(fcount)+'.jpg', crop)

        self.plotFeatures(frame, new_f, matches)

        # Recover boundary shift
        for i in range(new_f.shape[0]):
            new_f[i][0][0] = new_f[i][0][0] - self.edge
            new_f[i][0][1] = new_f[i][0][1] - self.edge
        
        self.f = new_f
        cv2.imwrite('../data/0701/2/stab/'+str(fcount)+'.jpg', frame)
        
    def plotFeatures(self, frame, new_f, matches):
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

        for [x0, y0, x1, y1] in matches:
            cv2.arrowedLine(frame, (x0, y0), (int(x0+10*(x1-x0)), int(y0+10*(y1-y0))),
                            (0,0,0), tipLength=1/np.sqrt((x1-x0)**2+(y1-y0)**2))
            #cv2.circle(frame, (x1,y1), 5, (0,255,255))

        cv2.arrowedLine(frame,
                        (int(self.window[1]), int(self.window[0])),
                        (int(self.window[1] + 10*self.dispField[0]), int(self.window[0] + 10*self.dispField[1])),
                        (0,255,255), thickness=3)
        

###FLOW###

video = Video()
img = cv2.imread('../data/0630/flat1/unstab/0.jpg')[int(video.window[0]-0.5*video.dist_height)+80:int(video.window[0]+0.5*video.dist_height)+80,
                                                    int(video.window[1]-0.5*video.dist_width):int(video.window[1]+0.5*video.dist_width), :]
cv2.imwrite('../data/0630/flat1/cut/0.jpg', img)
#video.matchFeaturesInit(img)

for i in range(1, 155):
    img = cv2.imread('../data/0630/flat1/unstab/'+str(i)+'.jpg')[int(video.window[0]-0.5*video.dist_height)+80:int(video.window[0]+0.5*video.dist_height)+80,
                                                             int(video.window[1]-0.5*video.dist_width):int(video.window[1]+0.5*video.dist_width), :]
    cv2.imwrite('../data/0630/flat1/cut/'+str(i)+'.jpg', img)
    #video.matchFeatures(img, i)


"""
K = np.matrix([[756.78768558,   0.        , 629.89805344],
                    [  0.        , 756.86336981, 345.49169401],
                    [  0.        ,   0.        ,   1.        ]])
d = 50
def cos(angle):
    return np.cos(np.deg2rad(angle))

def sin(angle):
    return np.sin(np.deg2rad(angle))

def findHomography(r, p, y, t):
    
    Rx = np.matrix([[  cos(r), -sin(r), 0.],
                    [  sin(r),  cos(r), 0.],
                    [      0.,      0., 1.]])
    
    Ry = np.matrix([[1.,       0.,      0.],
                    [0.,   cos(p), -sin(p)], 
                    [0.,   sin(p),  cos(p)]])


    Rz = np.matrix([[ cos(y), 0., sin(y)],
                    [     0., 1.,     0.],
                    [-sin(y), 0., cos(y)]])

    R = Rx*Ry*Rz
    n = [0., 0., 1.]
    
    H = K*(R + t*np.transpose(n)/d)*la.inv(K)
    return(H)

img = cv2.imread('../data/0701/2/unstab/97.jpg')
roll = -0.5
pitch = 0
H = findHomography(roll, pitch, 0., np.array([[0.], [0.], [0.]]))
warped = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
cv2.imwrite('../data/0701/2/unstab/97.jpg', warped)
"""
