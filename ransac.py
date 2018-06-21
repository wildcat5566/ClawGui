import numpy as np
import matplotlib.pyplot as plt

col = ['k', 'g', 'm', 'r']
tol = 1.25

def ransac_init(fullset, plot=False):
    plt.subplot(1,1,1)
    plt.plot(fullset[:,0], fullset[:,1], 'ko', markersize=5, markerfacecolor='None')
    plt.plot(fullset[:,2], fullset[:,3], 'ko', markersize=5)
    hypo_inliers = fullset[0:int(len(fullset) / 3)]

    ### Model fitting
    con_x = []
    con_y = []
    for [x0, y0, x1, y1] in hypo_inliers:
        con_x.append(x1-x0)
        con_y.append(y1-y0)
        
    model = [np.mean(con_x), np.mean(con_y), tol*np.std(con_x), tol*np.std(con_y)]

    if plot==True:
        print("#Model#")
        print("[x_mean, y_mean, x_std, y_std]:")
        print(model)
        for [x0, y0, x1, y1] in fullset:
            plt.arrow(x0, y0, (x1 - x0), (y1 - y0), fc="k", ec="k", head_width=10, head_length=10, width=1,length_includes_head=True)
        for [x0, y0, x1, y1] in hypo_inliers:
            plt.arrow(x0, y0, (x1 - x0), (y1 - y0), fc="g", ec="g", head_width=10, head_length=10, width=1,length_includes_head=True)
        plt.arrow(np.mean(fullset[:,0]), np.mean(fullset[:,1]), np.mean(con_x), np.mean(con_y),
                  fc=col[1], ec=col[1], head_width=10, head_length=10, width=2,length_includes_head=True)
    return model

def ransac(fullset, model, iteration, plot=False):
    con_x = []
    con_y = []
    for [x0, y0, x1, y1] in fullset:
        if (x1-x0) <= model[0] + model[2] and (x1-x0) >= model[0] - model[2] and (y1-y0) <= model[1] + model[3] and (y1-y0) >= model[1] - model[3]:
            con_x.append(x1 - x0)
            con_y.append(y1 - y0)
            #print([x1 - x0, y1 - y0])
            if plot==True:
                plt.arrow(x0, y0, x1-x0, y1-y0, fc=col[iteration], ec=col[iteration], head_width=10, head_length=10, width=1,length_includes_head=True)
                
    model = [np.mean(con_x), np.mean(con_y), tol*np.std(con_x), tol*np.std(con_y)]
    
    if plot==True:
        print("#Model#")
        print("[x_mean, y_mean, x_std, y_std]:")
        print(model)
        plt.arrow(np.mean(fullset[:,0]), np.mean(fullset[:,1]), model[0], model[1],
                  fc=col[iteration], ec=col[iteration], head_width=10, head_length=10, width=2,length_includes_head=True)
    return model
"""
###Test code###

matches = np.array([[681.0, 404.0, 683.0, 401.0], [463.0, 484.0, 466.0, 482.0], [650.0, 344.0, 653.0, 341.0], [678.0, 129.0, 695.0, 
150.0], [678.0, 129.0, 684.0, 96.0], [217.0, 248.0, 237.0, 268.0], [217.0, 248.0, 209.0, 215.0], [386.0, 460.0, 
385.0, 444.0], [497.0, 243.0, 499.0, 239.0], [130.0, 206.0, 158.0, 211.0], [554.0, 194.0, 556.0, 190.0], [232.0, 
328.0, 201.0, 340.0], [232.0, 328.0, 256.0, 326.0], [533.0, 455.0, 535.0, 452.0], [588.0, 364.0, 574.0, 357.0], 
[457.0, 121.0, 431.0, 105.0], [154.0, 128.0, 164.0, 107.0], [232.0, 120.0, 231.0, 118.0], [423.0, 197.0, 420.0, 
198.0], [31.0, 48.0, 30.0, 44.0], [501.0, 613.0, 472.0, 597.0], [283.0, 578.0, 288.0, 580.0], [309.0, 51.0, 
307.0, 42.0], [532.0, 396.0, 535.0, 393.0], [166.0, 50.0, 167.0, 46.0], [315.0, 401.0, 299.0, 404.0], [388.0, 
584.0, 411.0, 587.0]])

model = ransac_init(matches, plot=True)
for i in range(2,4):
    model = ransac(matches, model, i, plot=True) #hypo_outliers
plt.show()
"""
