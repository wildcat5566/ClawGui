import numpy as np
def ransac_init(fullset, tolerance):
    hypo_inliers = fullset[0:int(len(fullset) / 3)]
    ### Model fitting
    con_x = []
    con_y = []
    for [x0, y0, x1, y1] in hypo_inliers:
        con_x.append(x1-x0)
        con_y.append(y1-y0)
        
    model = [np.mean(con_x), np.mean(con_y), tolerance*np.std(con_x), tolerance*np.std(con_y)]
    return model

def ransac(fullset, model, iteration, tolerance):
    con_x = []
    con_y = []
    for [x0, y0, x1, y1] in fullset:
        if (x1-x0) <= model[0] + model[2] and (x1-x0) >= model[0] - model[2] and (y1-y0) <= model[1] + model[3] and (y1-y0) >= model[1] - model[3]:
            con_x.append(x1 - x0)
            con_y.append(y1 - y0)

    model = [np.mean(con_x), np.mean(con_y), tolerance*np.std(con_x), tolerance*np.std(con_y)]
    return model
