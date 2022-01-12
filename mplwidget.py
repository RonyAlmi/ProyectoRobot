# ------------------------------------------------------
# -------------------- mplwidget.py --------------------
# ------------------------------------------------------
from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class MplWidget(QWidget):

    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        self.figure = plt.figure()
        #self.style= plt.style.use('dark_background')

        #self.figure.patch.set_facecolor("black")
        self.canvas = FigureCanvas(self.figure)


        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)

        self.canvas.axes.spines['top'].set_visible(False)
        self.canvas.axes.spines['right'].set_visible(False)
        self.canvas.axes.spines['bottom'].set_visible(False)
        self.canvas.axes.spines['left'].set_visible(False)

        #self.canvas.limite=self.canvas.figure.subplots_adjust(hspace=.5)
        self.canvas.limite = self.canvas.figure.subplots_adjust(left=0, bottom=0, right=1, top=1)
        #self.canvas.axes.grid()

        self.setLayout(vertical_layout)

class MplWidget_2(QWidget):
    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        self.figure = plt.figure()
        #self.style= plt.style.use('dark_background')

        #self.figure.patch.set_facecolor("black")
        self.canvas = FigureCanvas(self.figure)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)

        self.canvas.axes.spines['top'].set_visible(False)
        self.canvas.axes.spines['right'].set_visible(False)
        self.canvas.axes.spines['bottom'].set_visible(False)
        self.canvas.axes.spines['left'].set_visible(False)