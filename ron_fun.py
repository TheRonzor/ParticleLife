import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

class EmotiRon():
    TAU = 2*np.pi

    HEAD_RES = 100

    EYE_RES = 50
    EYE_X = 0.35
    EYE_Y = 0.4
    EYE_RADIUS = 0.1

    NOSE_RES = 10
    NOSE_WIDTH = 0.1
    NOSE_X = 0
    NOSE_Y = 0

    SMILE_RES = 50
    SMILE_CENTER = [0.3, 0.5]
    SMILE_TRIM = 0.6

    CMAP_DEF = 'cividis'

    FIG_SIZE = [0.5,0.5]

    def __init__(self, 
                 how_bad    = 0,        # Input 0 to 1
                 cmap       = CMAP_DEF
                 ):
        self.how_bad = how_bad
        self.cmap = cmap
        self.make_data()
        return
    
    def make_data(self):
        self.make_eyes()
        self.make_nose()
        self.make_smile()
        return
    
    def get_circle(self, center, radius, num_points):
        theta = np.linspace(0, 2*self.TAU, num_points)
        x = center[0] + radius*np.cos(theta)
        y = center[1] + radius*np.sin(theta)
        return list(x), list(y)
    
    def make_eyes(self):
        x = []
        y = []
        left_eye = {'center'     : [-self.EYE_X, self.EYE_Y],
                    'radius'     : self.EYE_RADIUS,
                    'num_points' : self.EYE_RES}
        right_eye = {'center'    : [self.EYE_X, self.EYE_Y],
                    'radius'     : self.EYE_RADIUS,
                    'num_points' : self.EYE_RES}
        for eye in [left_eye, right_eye]:
            xi,yi = self.get_circle(eye['center'], eye['radius'], eye['num_points'])
            x += xi
            y += yi
        self.eyes = np.array([x, y])
        return
    
    def make_nose(self):
        x = np.linspace(-self.NOSE_WIDTH/2, self.NOSE_WIDTH/2, self.NOSE_RES) + self.NOSE_X
        y = np.abs(x)
        self.nose = np.array([list(x), list(y)])
        return
    
    def make_smile(self):
        theta = np.linspace(np.pi+self.SMILE_TRIM, 2*np.pi-self.SMILE_TRIM, self.SMILE_RES)
        x = np.cos(theta)/2
        y = np.sin(theta) + self.SMILE_CENTER[0]*self.how_bad + self.SMILE_CENTER[1]*(1-self.how_bad)
        
        my = np.mean(y)
        dy = y - my
        y = (my+dy)*self.how_bad + (my-dy)*(1-self.how_bad)
        self.smile = np.array([list(x), list(y)])
        return self.smile
    
    def get_color(self):
        colormap = get_cmap(self.cmap)
        return colormap(self.how_bad)
    
    def get_plot(self):
        self.fig, self.ax = plt.subplots(figsize=self.FIG_SIZE)
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.axis('off')

        self.face = self.ax.scatter(0,0, 
                                    s      = 7e2, 
                                    color  = self.get_color())
        
        self.eye_scat = self.ax.scatter(self.eyes[0], self.eyes[1], color='w', s=1)
        self.nose_scat = self.ax.scatter(self.nose[0], self.nose[1], color='w', s=1)
        self.smile_scat = self.ax.scatter(self.smile[0], self.smile[1], color='w', s=1)

        return self.fig, self.ax, self.face, self.smile_scat,

#class Mapper():
#    def __init__(self, xmin, xmax, ymin, ymax, res):
#        self.x = np.linspace(xmin, xmax, res)
#        self.y = np.linspace(ymin, ymax, res)
#        self.map = {xi:yi for xi, yi in zip(self.x, self.y)}
#        return