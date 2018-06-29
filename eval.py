import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Dataset evaluation code
def eval(N):
    unstab_means = []
    stab_means = []
    for i in range(N):
        unstab0 = cv2.imread("../data/0629/unstab/"+str(i)+".jpg")
        unstab1 = cv2.imread("../data/0629/unstab/"+str(i+1)+".jpg")
        stab0 = cv2.imread("../data/0629/stab/"+str(i)+".jpg")
        stab1 = cv2.imread("../data/0629/stab/"+str(i+1)+".jpg")    
    
        unstab0 = cv2.cvtColor(unstab0,cv2.COLOR_BGR2GRAY)
        unstab1 = cv2.cvtColor(unstab1,cv2.COLOR_BGR2GRAY)
        stab0 = cv2.cvtColor(stab0,cv2.COLOR_BGR2GRAY)
        stab1 = cv2.cvtColor(stab1,cv2.COLOR_BGR2GRAY)
    
        unstab_hsv = np.zeros_like(unstab0)
        stab_hsv = np.zeros_like(stab0)
    
        flow = cv2.calcOpticalFlowFarneback(unstab0, unstab1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        unstab_mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])

        flow = cv2.calcOpticalFlowFarneback(stab0, stab1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        stab_mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])

        #print(np.mean(unstab_mag), np.mean(stab_mag))
        unstab_means.append(np.median(unstab_mag))
        stab_means.append(np.median(stab_mag))

    ind = np.arange(N)
    width = 0.35
    bars1 = plt.bar(ind, unstab_means, width, color='tab:brown')
    bars2 = plt.bar(ind+width, stab_means, width, color='tab:orange')
    plt.title('Dense optical flow magnitude mean')
    plt.legend([bars1, bars2], ['Original','Stabilized'])
    plt.show()

# Demonstration plot
def demoplot(n):
    unstab0 = cv2.imread("../data/0629/unstab/"+str(n)+".jpg")
    unstab1 = cv2.imread("../data/0629/unstab/"+str(n+1)+".jpg")
    stab0 = cv2.imread("../data/0629/stab/"+str(n)+".jpg")
    stab1 = cv2.imread("../data/0629/stab/"+str(n+1)+".jpg")    
    unstab_hsv = np.zeros_like(unstab0)
    stab_hsv = np.zeros_like(stab0)
    
    unstab0 = cv2.cvtColor(unstab0,cv2.COLOR_BGR2GRAY)
    unstab1 = cv2.cvtColor(unstab1,cv2.COLOR_BGR2GRAY)
    stab0 = cv2.cvtColor(stab0,cv2.COLOR_BGR2GRAY)
    stab1 = cv2.cvtColor(stab1,cv2.COLOR_BGR2GRAY)

    unstab_hsv[...,0] = 0 #ang*180/np.pi/2 # Orientation, hue-encoded
    unstab_hsv[...,1] = 0 #Saturation
    stab_hsv[...,0] = 0
    stab_hsv[...,1] = 0 

    flow = cv2.calcOpticalFlowFarneback(unstab0, unstab1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])

    unstab_hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(unstab_hsv,cv2.COLOR_HSV2BGR)

    mm = np.mean(mag)
    mmax = np.max(mag)

    plt.subplot(121)
    plt.title("Original: Dense optical flow")
    plt.imshow(bgr)

    flow = cv2.calcOpticalFlowFarneback(stab0, stab1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    stab_hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(stab_hsv,cv2.COLOR_HSV2BGR)

    plt.subplot(122)
    plt.title("Stabilized: Dense optical flow")
    plt.imshow(bgr)

    plt.show()

# Video example code
def video_optflow():
    
    cap = cv2.VideoCapture(0)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,0] = 0
    hsv[...,1] = 0
    while(1):
        t = time.time()
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        print(time.time() - t)
        cv2.imshow('frame2',bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',bgr)
        prvs = next
    
    cap.release()
    cv2.destroyAllWindows()

def cleardir():
    N=len(os.listdir('../data/0629/unstab'))
    for i in range(N):
        os.remove('../data/0629/unstab/'+str(i)+'.jpg')
        os.remove('../data/0629/stab/'+str(i)+'.jpg')

def removeimg(n):
    os.remove('../data/0629/unstab/'+str(n)+'.jpg')
    os.remove('../data/0629/stab/'+str(n)+'.jpg')
    for i in range(n+1, len(os.listdir('../data/0629/stab'))):
        os.rename('../data/0629/unstab/'+str(i)+'.jpg', '../data/0629/unstab/'+str(i-1)+'.jpg')
        os.rename('../data/0629/stab/'+str(i)+'.jpg', '../data/0629/stab/'+str(i-1)+'.jpg')


N=len(os.listdir('../data/0629/stab'))
print(N)
eval(N-2)
#cleardir()
