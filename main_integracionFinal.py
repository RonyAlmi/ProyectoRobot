import sys
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import numpy as np
from PyQt5.QtWidgets import*
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon, QFont
from PyQt5 import QtWidgets, QtGui
from PyQt5 import QtCore
from PyQt5.Qt import Qt
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QImage , QPixmap
from PyQt5.QtWidgets import QDialog , QApplication
from PyQt5.uic import loadUi
import socket
import pygame
import time
from time import time
from openpyxl import Workbook
import math
import random
import matplotlib.pyplot as plt
# Functions
import Calibration as calib
import HSV_filter as hsv
import Centroide as cent
import Triangulation as tri
import hilo
from RobotArm import *
import Segment as sg
import RobotArm_simul as ra

drawnItems=ra.drawnItems
focusedJoint=ra.focusedJoint
forward=False
backward=False

visualClose = False
controlMouseClose = False

frame_left = None
frame_right = None
roiPts_left = []
roiPts_right = []
SelectPoint_rigth=False
SelectPoint_left=False
inputMode = False
high = 10
width = 5
flag_mouse=""

mode_cam = 1
cam_a = 1
cam_b = 2
cam_c = 3
size_a = 750
size_b = 360
size_c = 360

class gestual(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        loadUi("gestual.ui", self)

###############################################################
###############################################################

class visual(QDialog):
    def __init__(self):
        global value1, value2, visualClose
        QDialog.__init__(self)
        loadUi("visual.ui", self)
        #self.but_cerrar.clicked.connect(self.cerrar)
        self.manual_but2.clicked.connect(self.control_main)
        self.value1 = 640
        self.value2 = 640
        self.flag_cam = 0
        self.flag_mouse = 0
        #self.but_aceptar.clicked.connect(self.main_visual)
        #self.main_visual()

    def keyPressEvent(self, event):
        if event.text() == "i":
            self.flag_cam = 1

    def control_main(self):
        global visualClose
        #self.tehseencode.exec_()
        self.close()
        visualClose = True
        self.close()
        #self.tehseencode.exec_()
        #self.tehseencode.onClicked()

    def mouseReleaseEvent(self, event):
        global pause
        # pause=False
        print("mouse soltado")

    def mousePressEvent(self, event):
        global frame_left, frame_right, roiPts_right, roiPts_left, inputMode, pause, SelectPoint_left, SelectPoint_rigth
        if event.buttons() & QtCore.Qt.LeftButton:
            if inputMode and len(roiPts_left) < 1 and SelectPoint_left:
                x = event.x() -30
                y = event.y() - 38
                print(x)
                print(y)

                x_min = x - width
                x_max = x + width
                y_min = y - high
                y_max = y + high

                roiPts_left = (x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)

                cv2.circle(frame_left, (x, y), 5, (0, 0, 0), -1)
                for i in range(0, 4):
                    cv2.circle(frame_left, roiPts_left[i], 4, (255, 255, 0), 2)

                frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
                height1, width1, channel1 = frame_left.shape
                step1 = channel1 * width1
                qImg1 = QImage(frame_left.data, width1, height1, step1, QImage.Format_RGB888)
                img1 = qImg1.scaled(self.value1, self.value1, Qt.KeepAspectRatio)
                self.camara1_label.setAlignment(Qt.AlignCenter)
                self.camara1_label.setPixmap(QPixmap.fromImage(img1))
                SelectPoint_left = False

            if inputMode and len(roiPts_right) < 1 and SelectPoint_rigth:
                x = event.x() - 670
                y = event.y() - 38
                print(x)
                print(y)

                x_min = x - width
                x_max = x + width
                y_min = y - high
                y_max = y + high

                roiPts_right = (x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)

                cv2.circle(frame_right, (x, y), 5, (0, 0, 0), -1)
                for i in range(0, 4):
                    cv2.circle(frame_right, roiPts_right[i], 4, (255, 255, 0), 2)

                frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
                height2, width2, channel2 = frame_right.shape
                step2 = channel2 * width2
                qImg2 = QImage(frame_right.data, width2, height2, step2, QImage.Format_RGB888)
                img2 = qImg2.scaled(self.value2, self.value2, Qt.KeepAspectRatio)
                self.camara2_label.setAlignment(Qt.AlignCenter)
                self.camara2_label.setPixmap(QPixmap.fromImage(img2))
                # cv2.imshow("frame left", frame_left)
                # cv2.imshow("frame right", frame_right)
                key = cv2.waitKey(1) & 0xFF
                SelectPoint_rigth = False

    def main_visual(self):
        global frame_left, frame_right, roiPts_right, roiPts_left, inputMode, SelectPoint_left, SelectPoint_rigth, visualClose

        C_L = 0
        C_R = 0
        cap_left = cv2.VideoCapture(C_L, cv2.CAP_DSHOW)
        cap_right = cv2.VideoCapture(C_R, cv2.CAP_DSHOW)

        # cap_left = cv2.VideoCapture('rtsp://admin:bylogic1234@192.168.1.64/1')
        # cap_right = cv2.VideoCapture('rtsp://admin:bylogic1234@192.168.1.64/1')

        frame_rate = 120  # Máximo 120 fps
        B = 7.5  # Distancia entre cámaras [cm] 12.8
        f = 3.6  # Longitud focal del lente [mm]
        alpha = 90  # Campo visual horizontal [°]
        count = -1

        # CALIBRACION
        # cal = input("¿Desea calibrar las cámaras? [S] o [N]: ")
        # if cal == "s" or cal == "S":
        #    Right_Stereo_Map, Left_Stereo_Map = calib.undistorted(frame_left, frame_right)

        # Configuración del mouse
        # cv2.namedWindow("frame left")
        # cv2.setMouseCallback("frame left", self.selectROI_left)

        # Criterios para terminar el seguimiento
        termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1)
        roiBox_left = None
        roiBox_right = None

        wb = Workbook()
        ws = wb.active
        while cap_left.isOpened() and cap_right.isOpened():
            # start_time = time()
            count += 1

            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            frame_left = cv2.resize(frame_left, (0, 0), fx=1, fy=1)  # hd 2 1.5 #fhd 3 2.25
            frame_right = cv2.resize(frame_right, (0, 0), fx=1, fy=1)

            if visualClose == True:
                print(visualClose)
                break
            if frame_left is None or frame_right is None:
                cap_left = cv2.VideoCapture(0)
                cap_right = cv2.VideoCapture(0)
                ret_left, frame_left = cap_left.read()
                ret_right, frame_right = cap_right.read()
            else:

                """
                if cal == "s" or cal == "S":
                    frame_left = cv2.remap(frame_left, Left_Stereo_Map[0], Left_Stereo_Map[1],
                                           interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
                    frame_right = cv2.remap(frame_right, Right_Stereo_Map[0], Right_Stereo_Map[1],
                                            interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
                """

                if ret_left is False or ret_right is False:

                    break

                else:
                    if roiBox_left is not None:
                        # Conversión de BGR a HSV del frame
                        hsv_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2HSV)
                        backProj_left = cv2.calcBackProject([hsv_left], [0], roiHist_left, [0, 180], 1)

                        # Se utiliza la funcion CAMSHIFT de OpenCV
                        (r_left, roiBox_left) = cv2.CamShift(backProj_left, roiBox_left, termination)
                        pts_left = np.int0(cv2.boxPoints(r_left))
                        cv2.polylines(frame_left, [pts_left], True, (85, 255, 0), 2)

                        mask_left = hsv.add_HSV_filter(frame_left, pts_left)
                        center_left = cent.find_moments(frame_left, mask_left)

                    if roiBox_right is not None:
                        # Conversión de BGR a HSV del frame
                        hsv_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2HSV)
                        backProj_right = cv2.calcBackProject([hsv_right], [0], roiHist_right, [0, 180], 1)

                        # Se utiliza la funcion CAMSHIFT de OpenCV
                        (r_right, roiBox_right) = cv2.CamShift(backProj_right, roiBox_right, termination)
                        pts_right = np.int0(cv2.boxPoints(r_right))
                        cv2.polylines(frame_right, [pts_right], True, (85, 255, 0), 2)

                        mask_right = hsv.add_HSV_filter(frame_right, pts_right)
                        # hs2v = cv2.cvtColor(mask_left, cv2.COLOR_BGR2GRAY)
                        # cv2.imshow(" right", hs2v)
                        center_right = cent.find_moments(frame_right, mask_right)

                        depth = tri.find_depth(center_left, center_right, frame_left, frame_right, B, f, alpha)
                        # depth = (depth - 10) * 4
                        # depth = (0.00000002 * (depth**3)) - (0.00005 * (depth**2)) + (0.0615 * depth) + 10.595
                        # depth = (2.4772 * depth) - 12.212
                        depth = -(0.00006 * (depth ** 3)) + (0.0166 * (depth ** 2)) + (1.6689 * depth) + 0.1489
                        ws.append([depth])
                        # print("Distancia: " + str(depth) + " cm")

                        cv2.putText(frame_left, "DISTANCIA: " + str(round(depth, 3)) + " cm",
                                    (center_left[0] - 100, center_left[1] - 100),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(frame_right, "DISTANCIA: " + str(round(depth, 3)) + " cm",
                                    (center_right[0] - 100, center_right[1] - 100),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 0), 2)

                    if inputMode is False:
                        cv2.putText(frame_left, "PRESIONAR [i] PARA SELECCIONAR OBJETO", (45, 40),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 0), 2)

                        cv2.line(frame_right, (320, 0), (320, 480), (85, 255, 0), 1)
                        cv2.line(frame_right, (0, 240), (640, 240), (85, 255, 0), 1)
                        cv2.line(frame_left, (320, 0), (320, 480), (85, 255, 0), 1)
                        cv2.line(frame_left, (0, 240), (640, 240), (85, 255, 0), 1)
                        cv2.circle(frame_right, (320, 240), 25, (85, 255, 0), 1)
                        cv2.circle(frame_left, (320, 240), 25, (85, 255, 0), 1)

                    frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
                    height1, width1, channel1 = frame_left.shape
                    step1 = channel1 * width1
                    qImg1 = QImage(frame_left.data, width1, height1, step1, QImage.Format_RGB888)
                    img1 = qImg1.scaled(self.value1, self.value1, Qt.KeepAspectRatio)
                    self.camara1_label.setAlignment(Qt.AlignCenter)
                    self.camara1_label.setPixmap(QPixmap.fromImage(img1))

                    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
                    height2, width2, channel2 = frame_right.shape
                    step2 = channel2 * width2
                    qImg2 = QImage(frame_right.data, width2, height2, step2, QImage.Format_RGB888)
                    img2 = qImg2.scaled(self.value2, self.value2, Qt.KeepAspectRatio)
                    self.camara2_label.setAlignment(Qt.AlignCenter)
                    self.camara2_label.setPixmap(QPixmap.fromImage(img2))

                    # cv2.imshow("frame left", frame_left)
                    # cv2.imshow("frame right", frame_right)
                    key = cv2.waitKey(1) & 0xFF

                    # if key == ord("i") or key == ord("I") and len(roiPts_left) < 4:
                    if self.flag_cam == 1 and len(roiPts_left) < 4:
                        print("test_pres_i")
                        inputMode = True
                        SelectPoint_left = True
                        orig_left = frame_left.copy()
                        orig_right = frame_right.copy()

                        while len(roiPts_left) < 1:
                            # cv2.imshow("frame left", frame_left)
                            print("test_pres_i")
                            loop = QEventLoop()
                            QTimer.singleShot(500, loop.quit)
                            loop.exec_()

                        print("test_bucle")
                        # cv2.namedWindow("frame right")
                        # cv2.setMouseCallback("frame right", self.selectROI_right)
                        cv2.putText(frame_right, "SELECCIONAR NUEVAMENTE EL OBJETO", (75, 40),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 0), 2)  # 85, 255, 0

                        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
                        height1, width1, channel1 = frame_left.shape
                        step1 = channel1 * width1
                        qImg1 = QImage(frame_left.data, width1, height1, step1, QImage.Format_RGB888)
                        img1 = qImg1.scaled(self.value1, self.value1, Qt.KeepAspectRatio)
                        self.camara1_label.setAlignment(Qt.AlignCenter)
                        self.camara1_label.setPixmap(QPixmap.fromImage(img1))

                        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
                        height2, width2, channel2 = frame_right.shape
                        step2 = channel2 * width2
                        qImg2 = QImage(frame_right.data, width2, height2, step2, QImage.Format_RGB888)
                        img2 = qImg2.scaled(self.value2, self.value2, Qt.KeepAspectRatio)
                        self.camara2_label.setAlignment(Qt.AlignCenter)
                        self.camara2_label.setPixmap(QPixmap.fromImage(img2))
                        # cv2.imshow("frame left", frame_left)
                        # cv2.imshow("frame right", frame_right)
                        # cv2.waitKey(1) & 0xFF
                        SelectPoint_rigth = True
                        while len(roiPts_right) < 1:
                            loop = QEventLoop()
                            QTimer.singleShot(500, loop.quit)
                            loop.exec_()

                        print("fin bucle config")
                        # Determina los puntos delimitadores del ROI
                        roiPts_left = np.array(roiPts_left)
                        s_left = roiPts_left.sum(axis=1)
                        tl_left = roiPts_left[np.argmin(s_left)]
                        br_left = roiPts_left[np.argmax(s_left)]

                        roiPts_right = np.array(roiPts_right)
                        s_right = roiPts_right.sum(axis=1)
                        tl_right = roiPts_right[np.argmin(s_right)]
                        br_right = roiPts_right[np.argmax(s_right)]

                        # Conversión de BGR a HSV del objeto
                        roi_left = orig_left[tl_left[1]:br_left[1], tl_left[0]:br_left[0]]
                        roi_left = cv2.cvtColor(roi_left, cv2.COLOR_BGR2HSV)

                        roi_right = orig_right[tl_right[1]:br_right[1], tl_right[0]:br_right[0]]
                        roi_right = cv2.cvtColor(roi_right, cv2.COLOR_BGR2HSV)

                        # Calculo del histograma del objeto y su normalización
                        roiHist_left = cv2.calcHist([roi_left], [0], None, [16], [0, 180])
                        roiHist_left = cv2.normalize(roiHist_left, roiHist_left, 0, 255, cv2.NORM_MINMAX)
                        roiBox_left = (tl_left[0], tl_left[1], br_left[0], br_left[1])

                        roiHist_right = cv2.calcHist([roi_right], [0], None, [16], [0, 180])
                        roiHist_right = cv2.normalize(roiHist_right, roiHist_right, 0, 255, cv2.NORM_MINMAX)
                        roiBox_right = (tl_right[0], tl_right[1], br_right[0], br_right[1])

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # elapsed_time = (time() - start_time) * 1000
                # print("Tiempo transcurrido: %0.10f ms" % elapsed_time + "\n")
        wb.save("Data.xlsx")

        cap_right.release()
        cap_left.release()
        cv2.destroyAllWindows()

class manual(QDialog):
    global controlMouseClose
    def __init__(self):
        super(manual, self).__init__()
        #QMainWindow.__init__(self)
        loadUi("manual.ui", self)
        self.setWindowTitle("BRAZO ROBOTICO")
        self.salir.clicked.connect(self.go_main)
        #self.pushButton_signal.clicked.connect(self.update_graph)
        #self.update_graph()
        #self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))

    def go_main(self):
        global controlMouseClose
        controlMouseClose = True
        self.close()
        self.close()

    def update_graph(self):
        global Arm, target, reach, target, targetPt, controlMouseClose
        # Instantiate robot arm class.
        Arm = RobotArm2D()

        # Add desired number of joints/links to robot arm object.
        Arm.add_revolute_link(length=6, thetaInit=math.radians(120))
        Arm.add_revolute_link(length=4, thetaInit=math.radians(45))
        Arm.add_revolute_link(length=1, thetaInit=math.radians(55))
        Arm.update_joint_coords()

        # Initialize target coordinates to current end effector position.
        target = Arm.joints[:, [-1]]
        self.MplWidget.canvas.axes.clear()
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
            global target, targetPt, controlMouseClose

            if controlMouseClose==True:
                return
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
        """
        def on_key_press(event):
            '''Key press event to stop script execution if Enter is pressed,
                or toggle mode if Shift is pressed.
            '''
            global exitFlag, mode
            if event.key == 'enter':
                exitFlag = True
            elif event.key == 'shift':
                mode *= -1

        self.MplWidget.canvas.mpl_connect('key_press_event', on_key_press)
        """
        # Turn on interactive plotting and show plot.
        self.MplWidget.canvas.flush_events()
        self.MplWidget.canvas.draw()

        print('Select plot window and press Shift to toggle mode or press Enter to quit.')

        # Variable "t" is used for moving target mode.
        t = 0.
        while not exitFlag:
            if controlMouseClose==True:
                break
                #return
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

class tehseencode(QMainWindow):
    global visualClose
    def __init__(self):
        super(tehseencode,self).__init__()
        loadUi("gui_full3.ui", self)
        #self.setStyleSheet("background-color: black;")
        self.statusBar().showMessage("Bienvenid@")
        self.showMaximized()
        self.setWindowTitle("ROBOT UNSA")
        self.conect_label.setText('CONECTANDO')
        # self.conect_label.setStyleSheet("background-color: rgb(243, 244, 169); color: black")
        self.conect_label.setFont(QFont('Arial', 16))
        self.logic = 0
        self.value = 1
        self.flag = 0

        self.SHOW.clicked.connect(self.onClicked)

        self.gestual_but.clicked.connect(self.gestual)
        self.visual_but.clicked.connect(self.visual)
        self.manual_but.clicked.connect(self.manual)

        self.gestual = gestual()
        self.visual = visual()
        self.manual = manual()

        #self.CAPTURE.clicked.connect(self.CaptureClicked)
        self.imgLabel.setText("CARGANDO CAMARAS...")
        self.imgLabel.setFont(QFont('Arial', 20, QtGui.QFont.Black))
        self.imgLabel.setAlignment(Qt.AlignCenter)
        self.simulador_arm()
        self.onClicked()

        #self.Worker1 = Worker1()
        #self.Worker1.start()
    @pyqtSlot()
    def start_com(self):
        self.TEXT.setText('Presione "CAPTURA " Para capturar una  imagen')
        self.TEXT.setText("PRESIONE 'CÁMARA' Para conectarse con la camara.")
        self.TEXT.setFont(QFont('Arial', 10, QtGui.QFont.Black))
        ###### iniciando socket###
        # TCP_IP = '192.168.1.11'
        self.TCP_IP = '192.168.1.100'
        self.TCP_PORT = 5552
        self.BUFFER_SIZE = 1024
        MESSAGE = "$MECA,DD,00,A,1,0"  # mensaje de inicializacion
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.TCP_IP, self.TCP_PORT))
            s.send(MESSAGE.encode("ascii"))
            data = s.recv(self.BUFFER_SIZE)
            s.close()
            print("received data:", data)
            self.TEXT.setText('Comunicación inicializada')
            self.conect_label.setText('CONECTADO')
            self.conect_label.setStyleSheet("background-color: rgb(243, 244, 169); color: black")
            self.conect_label.setFont(QFont('Arial', 16, QtGui.QFont.Black))
        except:
            self.TEXT.setText('No se pudo conectar con el robot')

        ##########################
    def manual(self):
        global controlMouseClose
        self.TEXT.setText('CONTROL MANUAL ACTIVADO')
        self.manual.showMaximized()
        controlMouseClose = False
        self.manual.update_graph()
        self.manual.exec_()

    def gestual(self):
        self.TEXT.setText('CONTROL GESTUAL ACTIVADO')
        self.gestual.showMaximized()
        self.gestual.exec_()

    def visual(self):
        global visualClose
        self.TEXT.setText('CONTROL VIUAL ACTIVADO')
        self.visual.showMaximized()
        visualClose=False
        #self.visual.setModal(True)
        self.visual.main_visual()
        self.visual.exec_()

    def simulador_arm(self):
        global focusedJoint, drawnItems, segments
        ################# Robot Stuff #####################
        # Declare the segments the robot arm will contain
        # Can have more than this many segments.
        s1 = sg.Segment(1, 0)
        s2 = sg.Segment(1, 0)
        s3 = sg.Segment(0.3, 0)

        # Place segments into a list, this list is used to initialize the robot arm
        segments = [s1, s2, s3]
        # Declare the angle configurations of the arm.
        angleConfig = [1.1, -0.72, -0.8]

        targetPt, = self.MplWidget.canvas.axes.plot([], [], marker='o', c='r')
        endEff, = self.MplWidget.canvas.axes.plot([], [], marker='o', markerfacecolor='w', c='g', lw=2)
        armLine, = self.MplWidget.canvas.axes.plot([], [], marker='o', c='g', lw=2)

        # Set axis limits based on reach from root joint.
        self.MplWidget.canvas.axes.set_xlim(-5, 5)
        self.MplWidget.canvas.axes.set_ylim(-5, 5)

        # self.MplWidget.canvas.mpl_connect('key_press_event', self.on_key_press)
        r1 = ra.RobotArm(segments, angleConfig)
        self.drawArm()
        return

    def drawArm(self):
        global drawnItems
        pos = ra.RobotArm.getJointPositionsxy()
        # Draw the actual segments of the robot
        for i in range(0, len(pos) - 1):
            pos = ra.RobotArm.getJointPositionsxy()
            s = pos[i]
            s[0] = s[0]
            s[1] = s[1]
            e = pos[i + 1]

            e[0] = e[0]
            e[1] = e[1]

            s = ra.pixCoor(s[0], s[1], 640, 480)
            e = ra.pixCoor(e[0], e[1], 640, 480)
            drawnItems.append(self.create_line(s, e, 1, 2))
            pos = ra.RobotArm.getJointPositionsxy()
        for i in range(0, len(pos)):
            spot = pos[i]

            spot[0] = spot[0]
            spot[1] = spot[1]

            spot = ra.pixCoor(spot[0], spot[1], 640, 480)
            # drawnItems.append(create_oval(spot[0]-5,spot[1]-5,spot[0]+5,spot[1]+5,fill="black"))
        return

    def create_line(self, init, end, color, w):
        point1 = init
        point2 = end

        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]

        self.MplWidget.canvas.axes.set_xlim(-2.5, 2.5)
        self.MplWidget.canvas.axes.set_ylim(-2.5, 2.5)
        self.MplWidget.canvas.axes.plot(x_values, y_values, marker='o', markerfacecolor='w', c='g', lw=2)
        #self.MplWidget.canvas.flush_events()
        self.MplWidget.canvas.draw()

    def clearLines(self):
        global drawnItems
        self.MplWidget.canvas.axes.clear()
        drawItems = []

    def onClicked(self):
        global cap, cap2, cap3, url, url_1, url_2, url_3, focusedJoint, forward, backward
        self.start_com()
        url_1 = 'rtsp://admin:bylogic1234@192.168.1.153:554/Streaming/channels/101'
        url_2 = 'rtsp://admin:bylogic1234@192.168.1.153:554/Streaming/channels/201'
        url_3 = 'rtsp://admin:bylogic1234@192.168.1.153:554/Streaming/channels/301'
        # Abre las cámaras
        try:
            cap = cv2.VideoCapture(url_1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap2 = cv2.VideoCapture(url_2)
            cap2.set(cv2.CAP_PROP_FPS, 30)
            cap3 = cv2.VideoCapture(url_3)
            cap3.set(cv2.CAP_PROP_FPS, 30)
        except:
            self.TEXT.setText('No se reconoce alguna camara')

        MESSAGE = "$MECA,DD,00,A,1,0"
        ####### Initialise the pygame library #######
        try:
            pygame.init()
            # Connect to the first JoyStick
            j = pygame.joystick.Joystick(0)
            j.init()
            self.TEXT.setText('Joystick Inicializado : %s' % j.get_name())
            self.TEXT.setFont(QFont('Arial', 8, QtGui.QFont.Black))
            print('Initialized Joystick : %s' % j.get_name())
            self.analog_keys = {0: 0, 1: 0, 2: 0, 3: 0, 4: -1, 5: -1}
            #############################
            self.cam_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            self.teclado_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
        except:
            self.TEXT.setText('No se iniacializó joystick')

        # Envolverlo
        try:
            fresh = hilo.FreshestFrame(cap)
            fresh2 = hilo.FreshestFrame(cap2)
            fresh3 = hilo.FreshestFrame(cap3)
        except:
            print("bloqueo")

        cnt = 0
        cnt2 = 0
        cnt3 = 0
        velocidad = "5"
        angulo=0
        flag_angulo=0
        flag_fast = 0
        while True:
            try:
                cnt, img = fresh.read(seqnumber=cnt + 1)
                cnt2, img2 = fresh2.read(seqnumber=cnt2 + 1)
                cnt3, img3 = fresh3.read(seqnumber=cnt3 + 1)
                self.displayImage_1(img, cam_a, size_a)
                self.displayImage_1(img2, cam_b, size_b)
                self.displayImage_1(img3, cam_c, size_c)
                cv2.waitKey(1)
            except:
                print("close")
            ################## obtencion joystick ######################
            events = pygame.event.get()
            for event in events:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.TCP_IP, self.TCP_PORT))
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 0:
                        velocidad="0"
                        print("x -> stop")
                        self.TEXT.setText('Boton X presionado')
                    # LEFT = True
                    if event.button == 1:
                        print("circulo")
                        self.TEXT.setText('CIRCULO presionado')

                    if event.button == 2:
                        print("cuadrado")
                        self.TEXT.setText('CUADRADO presionado: Velocidad --> SLOW')
                        velocidad="5"
                        flag_fast = 0
                        self.vel_label.setText('SLOW')
                        self.vel_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
                        self.vel_label.setFont(QFont('Arial', 16, QtGui.QFont.Black))
                        self.vel_label.setAlignment(Qt.AlignCenter)

                    if event.button == 3:
                        print("triangulo")
                        self.TEXT.setText('TRIANGULO presionado Velocidad--> FAST')
                        velocidad="9"
                        flag_fast=1
                        self.vel_label.setText('FAST')
                        self.vel_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
                        self.vel_label.setFont(QFont('Arial', 16, QtGui.QFont.Black))
                        self.vel_label.setAlignment(Qt.AlignCenter)
                    if event.button == 4:
                        print("share")
                        self.TEXT.setText('Boton SHARE presionado')
                    # UP = True
                    if event.button == 5:
                        print("ps")
                        self.TEXT.setText('Boton PS presionado')
                    if event.button == 6:
                        print("options")
                        self.TEXT.setText('Boton OPTIONS presionado')
                    if event.button == 7:
                        print("boton analogico izquiero")
                        self.TEXT.setText('Boton ANALOGICO IZQUIERDO presionado')
                    if event.button == 8:
                        print("boton analogico derecho")
                        self.TEXT.setText('Boton ANALOGICO DERECHO presionado')
                    if event.button == 9:
                        print("L1")
                        self.TEXT.setText('Boton L1 presionado')
                    if event.button == 10:
                        print("R1")
                        self.TEXT.setText('Boton R1 presionado')
                ########################################################################################
                    if event.button == 11:
                        print("Flecha hacia arriba")
                        flag_angulo=1
                        forward=True
                        self.TEXT.setText('FLECHA ARRIBA: Direccion --> Adelante')
                        MESSAGE = "$MECA,DD,00,A,1,"+velocidad
                    if event.button == 12:
                        print("Flecha hacia abajo")
                        flag_angulo=2
                        backward=True
                        self.TEXT.setText('FLECHA Abajo: Direccion --> Atras')
                        MESSAGE = "$MECA,DD,00,A,0," +velocidad
                    if event.button == 13:
                        print("Flecha a la izquierda")
                        self.TEXT.setText('FLECHA IZQUIERDA: Direccion --> Izquierda')
                    if event.button == 14:
                        print("Flecha a la derecha")
                        self.TEXT.setText('FLECHA DERECHA: Direccion --> Derecha')
                    if event.button == 15:
                        print("Touchpad")
                        self.TEXT.setText('Boton TOUCHPAD presionado')

                if event.type == pygame.JOYBUTTONUP:
                    #print("Button Released")
                    forward=False
                    backward=False
                    flag_angulo = 0
                    MESSAGE = "$MECA,DD,00,A,1,0"
                    #self.TEXT.setText('Dejó de puilsar el botón seleccionado')
                print("enviado"+MESSAGE)
                s.send(MESSAGE.encode("ascii"))
                data = s.recv(self.BUFFER_SIZE).decode("utf-8","ignore")
                #data=int(data[14:18], 16)
                s.close()
                # HANDLES ANALOG INPUTS
                if event.type == pygame.JOYAXISMOTION:
                    self.analog_keys[event.axis] = event.value
                    # print(analog_keys)
                    # Horizontal Analog
                    if abs(self.analog_keys[0]) > .4:
                        if self.analog_keys[0] < -.7:
                            LEFT = True
                        else:
                            LEFT = False
                        if self.analog_keys[0] > .7:
                            RIGHT = True
                        else:
                            RIGHT = False
                    # Vertical Analog
                    if abs(self.analog_keys[1]) > .4:
                        if self.analog_keys[1] < -.7:
                            UP = True
                        else:
                            UP = False
                        if self.analog_keys[1] > .7:
                            DOWN = True
                        else:
                            DOWN = False
                    # Triggers
                    if self.analog_keys[4] > 0:  # Left trigger
                        x = 0
                    # color += 2
                    if self.analog_keys[5] > 0:  # Right Trigger
                        y = -1
                    # color -= 2
            if forward:
                print(forward)
                ra.RobotArm.modJointAngle(focusedJoint, np.pi / 180)
                self.clearLines()
                self.drawArm()
            if backward:
                ra.RobotArm.modJointAngle(focusedJoint, -np.pi / 180)
                self.clearLines()
                self.drawArm()

            if flag_angulo==1:
                if flag_fast == 1:
                    angulo = angulo + 5
                if flag_fast == 0:
                    angulo=angulo+1
                self.angulo_label.setText(str(angulo))
                self.angulo_label.setFont(QFont('Arial', 18, QtGui.QFont.Black))
                self.angulo_label.setAlignment(Qt.AlignCenter)
            if flag_angulo == 2:
                if flag_fast == 1:
                    angulo = angulo - 5
                if flag_fast == 0:
                    angulo=angulo-1
                self.angulo_label.setText(str(angulo))
                self.angulo_label.setFont(QFont('Arial', 18, QtGui.QFont.Black))
                self.angulo_label.setAlignment(Qt.AlignCenter)

        cap.release()
        cap2.release()
        cap3.release()
        cv2.destroyAllWindows()
    def keyReleaseEvent(self, event):
        global forward, backward
        forward = False
        backward = False

    def keyPressEvent(self,event):
        global cam_a, cam_b, cam_c, size_a, size_b, size_c, mode_cam, focusedJoint, segments, forward, backward
        if event.text() == "0" or event.text() == "1" or event.text() == "2":
            print("number of segs: " + str(len(segments)))
            print(event.text())
            if int(event.text()) >= len(segments):
                focusedJoint = 0
            else:
                focusedJoint = int(event.text())
        if event.text() == 'w':
            print("you pressed w")
            forward=True

        elif event.text() == 'r':
            print("you pressed r")
            backward = True
        else:
            print("That button does nothing")
        if event.text() == "7":
            self.TEXT.setText('tecla presionado: --> cambio de camara')
            if mode_cam == 1:
                mode_cam = 2
                cam_a = 3
                cam_b = 1
                cam_c = 2
                size_a = 360
                size_b = 750
                size_c = 360
                return
            if mode_cam == 2:
                mode_cam = 3
                cam_a = 2
                cam_b = 3
                cam_c = 1
                size_a = 360
                size_b = 360
                size_c = 750

                return
            if mode_cam == 3:
                mode_cam = 1
                cam_a = 1
                cam_b = 2
                cam_c = 3
                size_a = 750
                size_b = 360
                size_c = 360
                return

        if event.text() == "1":
            #self.label_estado.setText("Procesando datos...")
            self.oruga_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            #self.label_estado.setFont(QFont('Arial', 16))
            self.torre_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.codo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.hombro_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.brazo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.garra_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.foto_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.luz_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.salir_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.captura_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
        if event.text() == "2":
            #self.label_estado.setText("Procesando datos...")
            self.torre_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            #self.label_estado.setFont(QFont('Arial', 16))
            self.oruga_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.codo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.hombro_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.brazo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.garra_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.foto_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.luz_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.salir_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
        if event.text() == "3":
            #self.label_estado.setText("Procesando datos...")
            self.codo_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            #self.label_estado.setFont(QFont('Arial', 16))
            self.oruga_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.torre_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.hombro_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.brazo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.garra_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.foto_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.luz_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.salir_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
        if event.text() == "4":
            #self.label_estado.setText("Procesando datos...")
            self.hombro_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            #self.label_estado.setFont(QFont('Arial', 16))
            self.oruga_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.torre_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.codo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.brazo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.garra_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.foto_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.luz_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.salir_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
        if event.text() == "5":
            #self.label_estado.setText("Procesando datos...")
            self.brazo_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            #self.label_estado.setFont(QFont('Arial', 16))
            self.oruga_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.torre_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.codo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.hombro_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.garra_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.foto_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.luz_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.salir_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
        if event.text() == "6":
            #self.label_estado.setText("Procesando datos...")
            self.garra_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            #self.label_estado.setFont(QFont('Arial', 16))
            self.oruga_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.torre_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.codo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.hombro_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.brazo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.foto_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.luz_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.salir_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
        if event.text() == "8":
            #self.label_estado.setText("Procesando datos...")
            self.CaptureClicked()
            self.foto_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            self.captura_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            #self.label_estado.setFont(QFont('Arial', 16))
            self.oruga_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.torre_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.codo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.hombro_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.brazo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.garra_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.luz_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.salir_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
        if event.text() == "9":
            #self.label_estado.setText("Procesando datos...")
            self.luz_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            #self.label_estado.setFont(QFont('Arial', 16))
            self.oruga_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.torre_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.codo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.hombro_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.brazo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.garra_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.foto_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.salir_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
        if event.text() == "0":
            #self.label_estado.setText("Procesando datos...")
            self.salir_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            #self.label_estado.setFont(QFont('Arial', 16))
        if event.text() == "j":
            self.joy_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            self.teclado_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
        if event.text() == "s":
            self.vel_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")

            self.zoo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.config_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
        if event.text() == "t":
            self.teclado_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            self.joy_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")

        if event.text() == "z":
            self.zoom_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            self.zoo_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            self.vel_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.config_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            if event.text() == "h":
                self.value1=value1+5
            if event.text() == "l":
                self.value1=value1-5

        if event.text() == "c":
            self.config_label.setStyleSheet("background-color: rgb(255, 255, 0); color: black")
            self.vel_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")
            self.zoo_label.setStyleSheet("background-color: rgb(240, 240, 240); color: black")

    def CaptureClicked(self):
        self.logic=2

    def displayImage_1(self, frame, cam, size_cam):
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height1, width1, channel1 = frame.shape
            step1 = channel1 * width1
            qImg1 = QImage(frame.data, width1, height1, step1, QImage.Format_RGB888)
            img1 = qImg1.scaled(size_cam, size_cam, Qt.KeepAspectRatio)
            if cam == 1:
                self.imgLabel.setAlignment(Qt.AlignCenter)
                self.imgLabel.setPixmap(QPixmap.fromImage(img1))
            if cam == 2:
                self.label_cam2.setAlignment(Qt.AlignCenter)
                self.label_cam2.setPixmap(QPixmap.fromImage(img1))
            if cam == 3:
                self.label_cam3.setAlignment(Qt.AlignCenter)
                self.label_cam3.setPixmap(QPixmap.fromImage(img1))
        except:
            print("no frame")


app =  QApplication(sys.argv)
window=tehseencode()
window.show()
#window.showFullScreen() #showFullScreen()
try:
    sys.exit(app.exec_())
except:
    print('excitng')
