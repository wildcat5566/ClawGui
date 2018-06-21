import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
import ransac

edge = 20

def init(img1, features_count, features_res, features_dist):
    gray1 = np.float32(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))[edge:img1.shape[0]-edge, edge:img1.shape[1]-edge]
    f = cv2.goodFeaturesToTrack(gray1, features_count, features_res, features_dist)
    return f
    
def matchFeatures(img2, window, f1, dist_width, dist_height, features_count, features_res, features_dist, match_minimum_dist,
                  plot=True, img1=None, count=0, matchdir=None, cropdir=None):
    ###Timing start###
    t = time.time()
    
    gray2 = np.float32(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))[edge:img2.shape[0]-edge, edge:img2.shape[1]-edge]
    f2 = cv2.goodFeaturesToTrack(gray2, features_count, features_res, features_dist)

    #avoid boundaries
    for i in range(f1.shape[0]):
        f1[i][0][0] = f1[i][0][0] + edge
        f1[i][0][1] = f1[i][0][1] + edge

    for i in range(f2.shape[0]):
        f2[i][0][0] = f2[i][0][0] + edge
        f2[i][0][1] = f2[i][0][1] + edge
       
    matches = []
    for i in range(f1.shape[0]):
        for j in range(i, f2.shape[0]):
            x0 = f1[i][0][0]
            y0 = f1[i][0][1]
            x1 = f2[j][0][0]
            y1 = f2[j][0][1]
            dist2 = (x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0)
            if(dist2 < match_minimum_dist):
                matches.append([x0, y0, x1, y1])

    print(len(matches))
    model = ransac.ransac_init(np.array(matches))
    model = ransac.ransac(np.array(matches), model, 2)
    model = ransac.ransac(np.array(matches), model, 3)
    displacementField = model
    #print(model)
    
    window[0] = window[0] + displacementField[1]*1.0 #0617random1 0.8
    window[1] = window[1] + displacementField[0]*1.0

    elapsed = time.time() - t
    print(elapsed)
    ###Timing over###

    if(plot==True):
        #fig = plt.figure(figsize=(12,6), dpi=160)
        
        #ax1 = plt.subplot(121)
        #ax1.imshow(img1)
        #for i in range(f1.shape[0]):
        #    ax1.plot(f1[i][0][0], f1[i][0][1], 'o', color='r', markerfacecolor='w', markersize=5)
        fig = plt.figure(figsize=(8,8), dpi=160)
        ax2 = plt.subplot(111)
        ax2.imshow(img2)
        for i in range(f2.shape[0]):
            ax2.plot(f2[i][0][0], f2[i][0][1], 'o', color='r', markerfacecolor='w', markersize=5)

        for i in range(len(matches)):
            ax2.plot(matches[i][0], matches[i][1], 'o', color='r', markerfacecolor='r', markersize=5)
            ax2.arrow(matches[i][0], matches[i][1],  (matches[i][2]-matches[i][0])*5, (matches[i][3]-matches[i][1])*5,
                      fc="k", ec="k", head_width=10, head_length=10)

        ### crop ###
        print(window)
        tb = int(window[0]-0.5*dist_height)
        bb = int(window[0]+0.5*dist_height)
        lb = int(window[1]-0.5*dist_width)
        rb = int(window[1]+0.5*dist_width)

        # Check if exceeds boundary
        exc = 0
        if tb < 0:
            print('tb:' + str(tb))
            exc = 1
            window[0] = window[0] - tb
            bb = bb - tb
            tb = 0
            
        elif bb > img2.shape[0]:
            exc = 1
            print('bb:' + str(bb))
            window[0] = window[0] - (bb - img2.shape[0])
            bb = img2.shape[0]
            tb = bb - dist_height
            
        if lb < 0:
            exc = 1
            print('lb:' + str(lb))
            window[1] = window[1] - lb 
            rb = rb - lb
            lb = 0
            
        elif rb > img2.shape[1]:
            exc = 1
            print('rb:' + str(rb))
            window[1] = window[1] - (rb - img2.shape[1])
            rb = img2.shape[1]
            lb = rb - dist_width

        # Plot rectangle
        if exc == 1:
            ax2.plot([lb, rb], [tb, tb], 'r-')
            ax2.plot([lb, rb], [bb, bb], 'r-')
            ax2.plot([lb, lb], [tb, bb], 'r-')
            ax2.plot([rb, rb], [tb, bb], 'r-')
        else:
            
            ax2.plot([lb, rb], [tb, tb], 'c-')
            ax2.plot([lb, rb], [bb, bb], 'c-')
            ax2.plot([lb, lb], [tb, bb], 'c-')
            ax2.plot([rb, rb], [tb, bb], 'c-')
            
        # Plot big arrow
        ax2.arrow(window[1], window[0], displacementField[0]*5, displacementField[1]*5,
                  fc="r", ec="r", head_width=20, head_length=20, width=5)
        
        crop = img2[tb:bb, lb:rb, :]
        
        fig.savefig(matchdir + str(count) + '.jpg')
        cv2.imwrite((cropdir + str(count) +'.jpg'),crop)
        plt.close()

    for i in range(f2.shape[0]):
        f2[i][0][0] = f2[i][0][0] - edge
        f2[i][0][1] = f2[i][0][1] - edge

    return window, f2
    
