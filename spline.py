import numpy as np

class Spline():
    def __init__(self, t):
        self.t = t
        self.df1 = 0
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
        A[5][9] = (t-1)
        A[5][10] = (t-1)*(t-1)
        A[5][11] = (t-1)*(t-1)*(t-1)
    
        ### Equations 6 and 7: Boundary conditions ###
        #eqn 6: f0'(x0) = a01 = given (Starting velocity as previously returned)
        A[6][1] = 1

        #eqn 7: reduce acceleration at ending with a reduction ratio alpha.
        #       f"23(t3) = 2a22+6a23t = alpha*f"23(t2) = alpha*2a22, 2(1-alpha)a22+6a23t=0
        alpha = 1.0
        A[7][10] = 1 - alpha
        A[7][11] = 6*(t-1)

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

        self.A = A

    def findSpline(self, y0, y1, y2, y3):
        b = np.transpose([y0, y1, y1, y2, y2, y3, self.df1, 0, 0, 0, 0, 0])
        x = np.linalg.solve(self.A, b)
        # x[4] = a10, x[5] = a11, x[6] = a12, x[7] = a13
        self.df1 = x[5]
        tar = []
        for i in range(self.t):
            tar.append(x[4] + x[5]*i + x[6]*i*i + x[7]*i*i*i)

        return tar

"""
import matplotlib.pyplot as plt

s = Spline(10)
targets = s.findSpline(1,5,4,7)
plt.subplot(121)
plt.plot([0, 10, 20, 29], [1,5,4,7], 'ro')
plt.plot([10,11,12,13,14,15,16,17,18,19], targets, 'ko')

targets = s.findSpline(5,4,6.8, 6)
plt.subplot(122)
plt.plot([0, 10, 20, 29], [5,4,6.8, 6], 'ro')
plt.plot([10,11,12,13,14,15,16,17,18,19], targets, 'ko')
plt.show()
"""
