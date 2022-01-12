#Library
import cv2
import numpy as np
from time import time, sleep
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
inputMode = False
high = 10
width = 5

def selectROI_left(event, x, y, flags, param):
    global frame_left, frame_right, roiPts_right, roiPts_left, inputMode

    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts_left) < 1:
        x_min = x - width
        x_max = x + width
        y_min = y - high
        y_max = y + high

        roiPts_left = (x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)

        cv2.circle(frame_left, (x, y), 5, (0, 0, 0), -1)
        for i in range(0, 4):
            cv2.circle(frame_left, roiPts_left[i], 4, (255, 255, 0), 2)

        cv2.imshow("frame left", frame_left)

def selectROI_right(event, x, y, flags, param):
    global frame_left, frame_right, roiPts_right, roiPts_left, inputMode

    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts_right) < 1:
        x_min = x - width
        x_max = x + width
        y_min = y - high
        y_max = y + high

        roiPts_right = (x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)

        cv2.circle(frame_right, (x, y), 5, (0, 0, 0), -1)
        for i in range(0, 4):
            cv2.circle(frame_right, roiPts_right[i], 4, (255, 255, 0), 2)

        cv2.imshow("frame right", frame_right)

class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

def main():
    global frame_left, frame_right, roiPts_right, roiPts_left, inputMode

    C_L = 0
    C_R = 2

    cap_left = cv2.VideoCapture(C_L, cv2.CAP_DSHOW)
    cap_right = cv2.VideoCapture(C_R, cv2.CAP_DSHOW)

    #cap_left = cv2.VideoCapture('rtsp://admin:Bylogic1@192.168.1.64:554/1')
    #cap_right = cv2.VideoCapture('rtsp://admin:Bylogic1@192.168.1.64:554/1')

    frame_rate = 120  # Máximo 120 fps
    B = 7.5           # Distancia entre cámaras [cm] 12.8
    f = 3.6           # Longitud focal del lente [mm]
    alpha = 90        # Campo visual horizontal [°]
    count = -1

    # CALIBRACION
    cal = input("¿Desea calibrar las cámaras? [S] o [N]: ")
    if cal == "s" or cal == "S":
        Right_Stereo_Map, Left_Stereo_Map = calib.undistorted(frame_left, frame_right)

    # Configuración del mouse
    cv2.namedWindow("frame left")
    cv2.setMouseCallback("frame left", selectROI_left)

    # Critérios para terminar el seguimiento
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1)
    roiBox_left = None
    roiBox_right = None

    wb = Workbook()
    ws = wb.active

    # Parámetros de inicialización de Kalman
    #kf = cv2.KalmanFilter(4, 2)
    #kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    #kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    kfObj = KalmanFilter()

    while cap_left.isOpened() and cap_right.isOpened():
        start_time = time()
        count += 1

        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        frame_left = cv2.resize(frame_left, (0, 0), fx=1, fy=1) #hd 2 1.5 #fhd 3 2.25
        frame_right = cv2.resize(frame_right, (0, 0), fx=1, fy=1)

        if cal == "s" or cal == "S":
            frame_left = cv2.remap(frame_left, Left_Stereo_Map[0], Left_Stereo_Map[1],
                                  interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
            frame_right = cv2.remap(frame_right, Right_Stereo_Map[0], Right_Stereo_Map[1],
                                   interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

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
                predictedCoords_l = kfObj.Estimate(center_left[0], center_left[1])

            if roiBox_right is not None:
                # Conversión de BGR a HSV del frame
                hsv_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2HSV)
                backProj_right = cv2.calcBackProject([hsv_right], [0], roiHist_right, [0, 180], 1)

                # Se utiliza la funcion CAMSHIFT de OpenCV
                (r_right, roiBox_right) = cv2.CamShift(backProj_right, roiBox_right, termination)
                pts_right = np.int0(cv2.boxPoints(r_right))
                cv2.polylines(frame_right, [pts_right], True, (85, 255, 0), 2)

                mask_right = hsv.add_HSV_filter(frame_right, pts_right)
                #hs2v = cv2.cvtColor(mask_left, cv2.COLOR_BGR2GRAY)
                #cv2.imshow(" right", hs2v)
                center_right = cent.find_moments(frame_right, mask_right)
                predictedCoords_r = kfObj.Estimate(center_right[0], center_right[1])

                depth = tri.find_depth(center_left, center_right, frame_left, frame_right, B, f, alpha)
                #depth = (depth - 10) * 4
                #depth = (0.00000002 * (depth**3)) - (0.00005 * (depth**2)) + (0.0615 * depth) + 10.595
                #depth = (2.4772 * depth) - 12.212
                depth = -(0.00006 * (depth ** 3)) + (0.0166 * (depth ** 2)) + (1.6689 * depth) + 0.1489
                ws.append([depth])
                print("Distancia: " + str(depth) + " cm")

                cv2.putText(frame_left, "DISTANCIA: " + str(round(depth, 3)) + " cm",
                            (center_left[0]-100, center_left[1]-100),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame_right, "DISTANCIA: " + str(round(depth, 3)) + " cm",
                            (center_right[0]-100, center_right[1]-100),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 0), 2)
                cv2.circle(frame_left, (int(predictedCoords_l[0]), int(predictedCoords_l[1])), 5, (0, 0, 255), -1)
                cv2.circle(frame_right, (int(predictedCoords_r[0]), int(predictedCoords_r[1])), 50, [0, 255, 255], 2, 8)

            if inputMode is False:
                cv2.putText(frame_left, "PRESIONAR [i] PARA SELECCIONAR OBJETO", (45, 40),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 0), 2)

                cv2.line(frame_right, (320, 0), (320, 480), (85, 255, 0), 1)
                cv2.line(frame_right, (0, 240), (640, 240), (85, 255, 0), 1)
                cv2.line(frame_left, (320, 0), (320, 480), (85, 255, 0), 1)
                cv2.line(frame_left, (0, 240), (640, 240), (85, 255, 0), 1)
                cv2.circle(frame_right, (320, 240), 25, (85, 255, 0), 1)
                cv2.circle(frame_left, (320, 240), 25, (85, 255, 0), 1)

            cv2.imshow("frame left", frame_left)
            cv2.imshow("frame right", frame_right)
            #sleep(0.5)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("i") or key == ord("I") and len(roiPts_left) < 4:
                inputMode = True
                orig_left = frame_left.copy()
                orig_right = frame_right.copy()

                while len(roiPts_left) < 1:
                    cv2.imshow("frame left", frame_left)
                    cv2.waitKey(0)

                cv2.namedWindow("frame right")
                cv2.setMouseCallback("frame right", selectROI_right)
                cv2.putText(frame_right, "SELECCIONAR NUEVAMENTE EL OBJETO", (75, 40),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("frame left", frame_left)
                cv2.imshow("frame right", frame_right)
                cv2.waitKey(1) & 0xFF

                while len(roiPts_right) < 1:
                    cv2.imshow("frame right", frame_right)
                    cv2.waitKey(0)

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

            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed_time = (time() - start_time) * 1000
            print("Tiempo transcurrido: %0.10f ms" % elapsed_time + "\n")

    wb.save("Data.xlsx")

    cap_right.release()
    cap_left.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    # https://www.youtube.com/watch?v=t3LOey68Xpg