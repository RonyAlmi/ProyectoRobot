from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import numpy as np
import matplotlib.pyplot as plt

import Segment as sg
import RobotArm_simul as ra

drawnItems=ra.drawnItems
focusedJoint=ra.focusedJoint


class MatplotlibWidget(QDialog):
    def __init__(self):
        #global 
        super(MatplotlibWidget, self).__init__()
        loadUi("arm_simul.ui", self)
        self.setWindowTitle("BRAZO ROBOTICO simulado")
        self.salir.clicked.connect(self.update_graph)
        #self.update_graph()
        #self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))

    def update_graph(self):
        global focusedJoint, drawnItems, segments
        ################# Robot Stuff #####################
        #Declare the segments the robot arm will contain
        #Can have more than this many segments.
        s1 = sg.Segment(1,0)
        s2 = sg.Segment(1,0)
        s3 = sg.Segment(1,0)
        
        #Place segments into a list, this list is used to initialize the robot arm
        segments = [s1,s2,s3]
        #Declare the angle configurations of the arm.
        angleConfig = [0,0,0]

        targetPt, = self.MplWidget.canvas.axes.plot([], [], marker='o', c='r')
        endEff, = self.MplWidget.canvas.axes.plot([], [], marker='o', markerfacecolor='w', c='g', lw=2)
        armLine, = self.MplWidget.canvas.axes.plot([], [], marker='o', c='g', lw=2)
        
        # Set axis limits based on reach from root joint.
        self.MplWidget.canvas.axes.set_xlim(-5,5)
        self.MplWidget.canvas.axes.set_ylim(-5,5)
        
        #self.MplWidget.canvas.mpl_connect('key_press_event', self.on_key_press)
        r1 = ra.RobotArm(segments,angleConfig)
        self.drawArm()
        return
    
    def keyPressEvent(self, event):
        global focusedJoint, segments
        if event.text() == "0" or event.text() == "1" or event.text() == "2":
            print ("number of segs: " + str(len(segments)))
            print (event.text())
            if int(event.text()) >= len(segments):
                focusedJoint = 0
            else:  
                focusedJoint = int(event.text())
        if event.text() == 'w':
            print ("you pressed w")
            ra.RobotArm.modJointAngle(focusedJoint, np.pi/180)
            self.clearLines()
            self.drawArm()
                   
        elif event.text() == 'r':
            print ("you pressed r")
            ra.RobotArm.modJointAngle(focusedJoint,-np.pi/180)
            self.clearLines()
            self.drawArm()
        else:
            print ("That button does nothing")
            
    def drawArm(self):
        global drawnItems 
        pos = ra.RobotArm.getJointPositionsxy()
        #Draw the actual segments of the robot
        for i in range(0,len(pos)-1):
            pos = ra.RobotArm.getJointPositionsxy()
            s = pos[i]
            s[0] = s[0]
            s[1] = s[1]
            e = pos[i+1]
 
            e[0] = e[0]
            e[1] = e[1]
                
            s = ra.pixCoor(s[0],s[1],640,480)
            e = ra.pixCoor(e[0],e[1],640,480)
            drawnItems.append(self.create_line(s, e, 1, 2))
            pos = ra.RobotArm.getJointPositionsxy()    
        for i in range(0,len(pos)):
            spot = pos[i]
                
            spot[0] = spot[0]
            spot[1] = spot[1]       
                
            spot = ra.pixCoor(spot[0],spot[1],640,480)
            #drawnItems.append(create_oval(spot[0]-5,spot[1]-5,spot[0]+5,spot[1]+5,fill="black"))
        return

    def create_line(self, init, end, color, w ):
        point1 = init
        point2 = end
        
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        
        self.MplWidget.canvas.axes.set_xlim(-4, 4)
        self.MplWidget.canvas.axes.set_ylim(-4, 4)
        self.MplWidget.canvas.axes.plot(x_values, y_values)
        self.MplWidget.canvas.flush_events()
        self.MplWidget.canvas.draw()

    def clearLines(self):
        global drawnItems, ax
        self.MplWidget.canvas.axes.clear()
        drawItems = [] 

app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()
