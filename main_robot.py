# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import math
from RobotArm import *


class MatplotlibWidget(QDialog):
    global Arm, target, reach, target, targetPt
    def __init__(self):

        super(MatplotlibWidget, self).__init__()
        #QMainWindow.__init__(self)
        loadUi("manual.ui", self)
        self.setWindowTitle("BRAZO ROBOTICO")
        self.salir.clicked.connect(self.update_graph)
        #self.update_graph()
        #self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))

    def update_graph(self):
        global Arm, target, reach, target, targetPt
        # Instantiate robot arm class.
        Arm = RobotArm2D()

        # Add desired number of joints/links to robot arm object.
        Arm.add_revolute_link(length=6, thetaInit=math.radians(120))
        Arm.add_revolute_link(length=4, thetaInit=math.radians(45))
        Arm.add_revolute_link(length=1, thetaInit=math.radians(55))
        Arm.update_joint_coords()

        # Initialize target coordinates to current end effector position.
        target = Arm.joints[:, [-1]]

        # Initialize plot and line objects for target, end effector, and arm.
        targetPt, = self.MplWidget.canvas.axes.plot([], [], marker='o', c='r')
        endEff, = self.MplWidget.canvas.axes.plot([], [], marker='o', markerfacecolor='w', c='g', lw=2)
        armLine, = self.MplWidget.canvas.axes.plot([], [], marker='o', c='g', lw=2)

        # Determine maximum reach of arm.
        reach = sum(Arm.lengths)

        # Set axis limits based on reach from root joint.
        self.MplWidget.canvas.axes.set_xlim(Arm.xRoot - 1.2 * reach, Arm.xRoot + 1.2 * reach)
        self.MplWidget.canvas.axes.set_ylim(Arm.yRoot - 1.2 * reach, Arm.yRoot + 1.2 * reach)

        # Add dashed circle to plot indicating reach.
        # circle = plt.Circle((Arm.xRoot, Arm.yRoot), reach, ls='dashed', fill=False)
        # ax.add_artist(circle)

        def update_plot():
            '''Update arm and end effector line objects with current x and y
                coordinates from arm object.
            '''
            armLine.set_data(Arm.joints[0, :-1], Arm.joints[1, :-1])
            endEff.set_data(Arm.joints[0, -2::], Arm.joints[1, -2::])

        update_plot()

        def move_to_target():
            '''Run Jacobian inverse routine to move end effector toward target.'''
            global Arm, target, reach

            # Set distance to move end effector toward target per algorithm iteration.
            distPerUpdate = 0.02 * reach
            if np.linalg.norm(target - Arm.joints[:, [-1]]) > 0.02 * reach:

                targetVector = (target - Arm.joints[:, [-1]])[:3]
                targetUnitVector = targetVector / np.linalg.norm(targetVector)
                deltaR = distPerUpdate * targetUnitVector
                J = Arm.get_jacobian()
                JInv = np.linalg.pinv(J)
                deltaTheta = JInv.dot(deltaR)
                Arm.update_theta(deltaTheta)
                Arm.update_joint_coords()
                update_plot()

        # "mode" can be toggled with the Shift key between 1 (click to set
        # target location) and -1 (target moves in predefined motion).
        mode = 1
        def on_button_press(event):
            '''Mouse button press event to set target at the location in the
                plot where the left mousebutton is clicked.
            '''
            global target, targetPt
            xClick = event.xdata
            yClick = event.ydata
            # print(xClick)
            # print(yClick)

            if (yClick < 0 or xClick > 0):
                print("NO SE ENCUENTRA EN EL AREA DE TRABAJO DEL ROBOT")

            # Ensure that the x and y click coordinates are within the axis limits
            # by checking that they are floats.
            if (mode == 1 and event.button == 1 and isinstance(xClick, float)
                    and isinstance(yClick, float) and yClick > 0 and xClick < 0):
                targetPt.set_data(xClick, yClick)
                target = np.array([[xClick, yClick, 0, 1]]).T

        self.MplWidget.canvas.mpl_connect('button_press_event', on_button_press)
        # Use "exitFlag" to halt while loop execution and terminate script.
        exitFlag = False

        def on_key_press(event):
            '''Key press event to stop script execution if Enter is pressed,
                or toggle mode if Shift is pressed.
            '''
            global exitFlag, mode
            print("ddddddd")
            if event.key == 'enter':
                exitFlag = True
            elif event.key == 'shift':
                mode *= -1

        self.MplWidget.canvas.mpl_connect('key_press_event', on_key_press)
        # Turn on interactive plotting and show plot.
        self.MplWidget.canvas.flush_events()
        self.MplWidget.canvas.draw()

        print('Select plot window and press Shift to toggle mode or press Enter to quit.')

        # Variable "t" is used for moving target mode.
        t = 0.
        while not exitFlag:

            if mode == -1:
                targetX = Arm.xRoot + 1.1 * (math.cos(0.12*t) * reach) * math.cos(t)
                targetY = Arm.yRoot + 1.1 * (math.cos(0.2*t) * reach) * math.sin(t)
                targetPt.set_data(targetX, targetY)
                target = np.array([[targetX, targetY, 0, 1]]).T
                t += 0.025
                #print("NO SE ENCUENTRA EN EL AREA DE TRABAJO DEL ROBOT")
            move_to_target()
            self.MplWidget.canvas.flush_events()
            self.MplWidget.canvas.draw()




app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()
