import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

class Spline():
    t = 1
    A = np.zeros((12, 12),dtype=float)

    ### Equations 0 to 5: Sampled measurements ###
    #eqn 0: f0(x0) = y0 = a00
    A[0][0] = 1

    #eqn 1: f0(x1) = y1 = a00+a01t+a02t^2+a03t^3
    A[1][0] = 1
    A[1][1] = t
    A[1][2] = t*t
    A[1][3] = t*t*t

    #eqn 2: f1(x1) = y1 = a10
    A[2][4] = 1

    #eqn 3: f2(x2) = y2 = a10+a11t+a12t^2+a13t^3
    A[3][4] = 1
    A[3][5] = t
    A[3][6] = t*t
    A[3][7] = t*t*t

    #eqn 4: f2(x2) = y2 = a20
    A[4][8] = 1

    #eqn 5: f2(x3) = y3 = a20+a21t+a22t^2+a23t^3
    A[5][8] = 1
    A[5][9] = t
    A[5][10] = t*t
    A[5][11] = t*t*t
    
    ### Equations 6 and 7: Boundary conditions ###
    #eqn 6: f0'(x0) = a01 = given (Starting velocity as previously returned)
    A[6][1] = 1

    #eqn 7: reduce acceleration at ending with a reduction ratio alpha.
    #       f"23(t3) = 2a22+6a23t = alpha*f"23(t2) = alpha*2a22, 2(1-alpha)a22+6a23t=0
    alpha = 1.0
    A[7][10] = 1 - alpha
    A[7][11] = 6*t

    ### Equation 8 and 9: 1st order derivatives continuity ###
    #eqn 8: f0'(x1) = f1'(x1), a01+2a02t+3a03t^2 = a11
    A[8][1] = 1
    A[8][2] = 2*t
    A[8][3] = 3*t*t
    A[8][5] = -1

    #eqn 9: f1'(x2) = f2'(x2), a11+2a12t+3a13t^2 = a21
    A[9][5] = 1
    A[9][6] = 2*t
    A[9][7] = 3*t*t
    A[9][9] = -1

    ### Equation 10 and 11: 2nd order derivatives continuity ###
    #eqn 10: f0"(x1) = f1"(x1), 2a02+6a03t=2a12
    A[10][2] = 2
    A[10][3] = 6*t
    A[10][6] = -2

    #eqn 11: f1"(x2) = f2"(x2), 2a12+6a13t=2a22
    A[11][6] = 2
    A[11][7] = 6*t
    A[11][10] = -2

    def __init__(self):
        self.b = np.zeros((12, 1),dtype=float)
        self.x = np.zeros((12, 1),dtype=float)

    def findSpline(self, y0, y1, y2, y3, df, iterations):
        self.b = np.transpose([y0, y1, y1, y2, y2, y3,  df, 0, 0, 0, 0, 0])
        self.x = la.solve(self.A, self.b)

        for i in range(3):
            xrange = np.arange(iterations + i*self.t, iterations + (i+1)*self.t, 0.01)
            yrange = []
            for j in range(len(xrange)):
                n = xrange[j]-xrange[0]
                yrange.append(self.x[4*i] + n*self.x[4*i+1] + n*n*self.x[4*i+2] + n*n*n*self.x[4*i+3])
            if(i==1):
                plt.plot(xrange, yrange, 'r')
            else:
                plt.plot(xrange, yrange, 'k:')
            
        # Return 1st order derivative (velocity) at x1 junction point
        # As new starting boundary conditions for next iteration
        # 1st order derivative: f1'(x1) = a11
        return self.x[5]

    def mapSpline(self, dx):
        return self.x[4] + dx*self.x[5] + dx*dx*self.x[6] + dx*dx*dx*self.x[7]

    def findWarpTargets(self, sampling_ratio, data_seq):
        pivots = [data_seq[0]]
        frames = len(data_seq)
        pivots_count = int(frames / sampling_ratio)
        x_timestamps = np.arange(0, pivots_count, 1 / sampling_ratio)
        for i in range(1, pivots_count):
            # Neighbor-averaging: anti-outlier
            pivots.append((data_seq[i * sampling_ratio - 1] + data_seq[i * sampling_ratio] + data_seq[i * sampling_ratio + 1]) / 3)

        targets = data_seq[0:sampling_ratio]
        df = 0 #Previous velocity as current starting boundary condition
        for i in range(pivots_count - 3):
            df = Spline.findSpline(self, pivots[i], pivots[i+1], pivots[i+2], pivots[i+3], df, i)
            targets.append(pivots[i+1])
            
            for j in range(sampling_ratio - 1):
                targets.append(Spline.mapSpline(self, x_timestamps[i * sampling_ratio + j + 1] - i))

        return targets
