import sys
import numpy as np

from PyQt5.QtCore import *
import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal
from time import time
from openpyxl import Workbook

# Functions
import Calibration as calib
import HSV_filter as hsv
import Centroide as cent
import Triangulation as tri

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

class tehseencode(QDialog):

    def __init__(self):
        super(tehseencode, self).__init__()
        # loadUi("student3.ui",self)
        loadUi("visual.ui", self)

        global value1, value2
        self.value1 = 480
        self.value2 = 480
        self.flag_cam = 0
        self.flag_mouse = 0
        self.but_aceptar.clicked.connect(self.main_visual)
        #self.main_visual()

    def keyPressEvent(self, event):
        if event.text() == "i":
            self.flag_cam = 1

    def mouseReleaseEvent(self, event):
        global pause
        #pause=False
        print("mouse soltado")

    def mousePressEvent(self, event):
        global frame_left, frame_right, roiPts_right, roiPts_left, inputMode, pause, SelectPoint_left, SelectPoint_rigth
        if event.buttons() & QtCore.Qt.LeftButton:
            if inputMode and len(roiPts_left) < 1 and SelectPoint_left:
                x= event.x()+70
                y= event.y()-50
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
                print("mouse _test 2222")
                x = event.x()-410
                y = event.y()-50
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
                #cv2.imshow("frame left", frame_left)
                #cv2.imshow("frame right", frame_right)
                key = cv2.waitKey(1) & 0xFF
                SelectPoint_rigth = False

    def main_visual(self):
        global frame_left, frame_right, roiPts_right, roiPts_left, inputMode, SelectPoint_left, SelectPoint_rigth

        C_L = 0
        C_R = 0

        # C_L = 0
        # C_R = 2

        cap_left = cv2.VideoCapture(C_L, cv2.CAP_DSHOW)
        cap_right = cv2.VideoCapture(C_R, cv2.CAP_DSHOW)

        #cap_left = cv2.VideoCapture('rtsp://admin:bylogic1234@192.168.1.64/1')
        #cap_right = cv2.VideoCapture('rtsp://admin:bylogic1234@192.168.1.64/1')

        frame_rate = 120  # Máximo 120 fps
        B = 7.5  # Distancia entre cámaras [cm] 12.8
        f = 3.6  # Longitud focal del lente [mm]
        alpha = 90  # Campo visual horizontal [°]
        count = -1

        # CALIBRACION
        #cal = input("¿Desea calibrar las cámaras? [S] o [N]: ")
        #if cal == "s" or cal == "S":
        #    Right_Stereo_Map, Left_Stereo_Map = calib.undistorted(frame_left, frame_right)

        # Configuración del mouse
        #cv2.namedWindow("frame left")
        #cv2.setMouseCallback("frame left", self.selectROI_left)

        # Criterios para terminar el seguimiento
        termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1)
        roiBox_left = None
        roiBox_right = None

        wb = Workbook()
        ws = wb.active

        while cap_left.isOpened() and cap_right.isOpened():
            #start_time = time()
            count += 1

            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            frame_left = cv2.resize(frame_left, (0, 0), fx=1, fy=1)  # hd 2 1.5 #fhd 3 2.25
            frame_right = cv2.resize(frame_right, (0, 0), fx=1, fy=1)

            if frame_left is None or frame_right is None:
                cap_left = cv2.VideoCapture('rtsp://admin:bylogic1234@192.168.1.64/1')
                cap_right = cv2.VideoCapture('rtsp://admin:bylogic1234@192.168.1.64/1')
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
                        #print("Distancia: " + str(depth) + " cm")

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

                    #cv2.imshow("frame left", frame_left)
                    #cv2.imshow("frame right", frame_right)
                    key = cv2.waitKey(1) & 0xFF

                    #if key == ord("i") or key == ord("I") and len(roiPts_left) < 4:
                    if self.flag_cam == 1 and len(roiPts_left) < 4:
                        inputMode = True
                        SelectPoint_left = True
                        orig_left = frame_left.copy()
                        orig_right = frame_right.copy()

                        while len(roiPts_left) < 1:
                            #cv2.imshow("frame left", frame_left)
                            loop = QEventLoop()
                            QTimer.singleShot(500, loop.quit)
                            loop.exec_()

                        print("test_bucle")
                        #cv2.namedWindow("frame right")
                        #cv2.setMouseCallback("frame right", self.selectROI_right)
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
                        #cv2.imshow("frame left", frame_left)
                        #cv2.imshow("frame right", frame_right)
                        #cv2.waitKey(1) & 0xFF
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

                    #elapsed_time = (time() - start_time) * 1000
                    #print("Tiempo transcurrido: %0.10f ms" % elapsed_time + "\n")

        wb.save("Data.xlsx")

        cap_right.release()
        cap_left.release()
        cv2.destroyAllWindows()


app = QApplication(sys.argv)
window = tehseencode()
window.show()
try:
    sys.exit(app.exec_())
except:
    print('excitng')