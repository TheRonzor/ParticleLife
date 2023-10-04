import numpy as np
from PyQt5 import QtGui as qtg
from PyQt5.QtCore import Qt as qt
from PyQt5 import QtWidgets as qtw

class RonSliderBase():
    '''Settings behind a QSlider'''
    def __init__(self, 
                 vmin, 
                 vmax, 
                 vdef, 
                 vsteps
                 ):
        self.values = np.linspace(vmin,vmax,vsteps)
        self.map = {n:v for n,v in enumerate(self.values)}
        self.vmin = 0
        self.vmax = len(self.map)-1
        self.vdef = self.get_key(vdef)
        return
    def get_key(self, value):
        return np.abs(self.values-value).argmin()
    def get_closest_value(self, value):
        return self.map[self.get_key(value)]
    
class RonSlider(RonSliderBase):
    '''A horizontal or vertical slider, based on continuous inputs for vmin, vmax, etc'''

    # Should change the inheritance structure so the QSlider is the parent

    def __init__(self,
                 vmin,
                 vmax,
                 vdef,
                 vsteps,
                 orient='horizontal'
                 ):
        super().__init__(vmin, vmax, vdef, vsteps)
        if orient=='horizontal':
            self.slider = qtw.QSlider(qt.Horizontal)
        elif orient=='vertical':
            self.slider = qtw.QSlider(qt.Vertical)
        else:
            raise ValueError("Orientation must be 'horizontal' or 'vertical'.'" 
                             + str(orient) + "' is not recognized.")
        
        self.slider.setMinimum(self.vmin)
        self.slider.setMaximum(self.vmax)
        self.slider.setValue(self.vdef)
        self.slider.setSingleStep(1)
        self.valueChanged = self.slider.valueChanged

    def get_value(self):
        return self.map[self.slider.value()]
    def set_value(self, value):
        self.slider.setValue(self.get_key(value))
        return

class HSliderWithLabels(qtw.QHBoxLayout):
    '''A horizontal slider with name on the left and current value on the right'''
    def __init__(self, name, vmin, vmax, vdef, vsteps, display_prec = None):
        super().__init__()
        self.slider = RonSlider(vmin, vmax, vdef, vsteps, 'horizontal')

        if display_prec is None:
            display_prec = int(np.log10(vsteps))
        self.fmt = '.' + str(display_prec) + 'f'

        self.label = qtw.QLabel(name)
        self.display = qtw.QLabel(format(self.slider.get_value(), self.fmt))
        self.slider.valueChanged.connect(self.update_display)

        self.addWidget(self.label)
        self.addWidget(self.slider.slider)
        self.addWidget(self.display)

        self.get_value = self.slider.get_value
        self.set_value = self.slider.set_value
        return
    
    def update_display(self):
        self.display.setText(format(self.slider.get_value(), self.fmt))
        return
    
class HSliderLabelAbove(qtw.QVBoxLayout):
    '''A vertical slider with the current value centered above'''
    def __init__(self, vmin, vmax, vdef, vsteps, display_prec = None):
        super().__init__()
        self.slider = RonSlider(vmin, vmax, vdef, vsteps, 'horizontal')

        if display_prec is None:
            display_prec = int(np.log10(vsteps))
        self.fmt = '.' + str(display_prec) + 'f'

        self.display = qtw.QLabel(format(self.slider.get_value(), self.fmt))
        self.display.setAlignment(qt.AlignCenter)
        self.slider.valueChanged.connect(self.UpdateDisplay)

        self.addWidget(self.display)
        self.addWidget(self.slider.slider)

        self.get_value = self.slider.get_value
        self.set_value = self.slider.set_value
        return
    def UpdateDisplay(self):
        self.display.setText(format(self.slider.get_value(), self.fmt))
        return

class LabeledInput(qtw.QHBoxLayout):
    def __init__(self, 
                 name, 
                 default
                 ):
        super().__init__()

        label = qtw.QLabel(name)
        self.value = qtw.QLineEdit(str(default))
        self.value.setValidator(qtg.QIntValidator()) 

        self.addWidget(label)
        self.addWidget(self.value)

        return
    
    def get_value(self):
        return self.value.text()
    
    def set_value(self, value):
        self.value.setText(str(value))