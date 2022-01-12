# https://www.youtube.com/watch?v=t3LOey68Xpg

import sys
import numpy as np
import time
import imutils
import cv2
import glob

def undistorted(frameL, frameR):
    # Criterios de terminacion
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Preparar puntos de objeto
    objp = np.zeros((7 * 7, 3), np.float32)  # (9*6,3) tablero OpenCV
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)  # [0:9,0:6] tablero OpenCV

    objpoints = []  # Matriz de puntos 3D en el espacio del mundo real
    imgpointsR = []  # Matriz de puntos 2D en el plano de la imagen
    imgpointsL = []

    print('Inicio de la calibración ...')

    # Llama a las imágenes guardadas
    numero = int((len(glob.glob('*.png')) - 2) / 2)
    for i in range(0, numero):
        t = str(i)
        ChessImaR = cv2.imread('chessboard-R' + t + '.png', 0)
        ChessImaL = cv2.imread('chessboard-L' + t + '.png', 0)

        retR, cornersR = cv2.findChessboardCorners(ChessImaR, (7, 7), None)  # Numero de esquinas a buscar
        retL, cornersL = cv2.findChessboardCorners(ChessImaL, (7, 7), None)  # (9, 6) tablero OpenCV

        if (retR == True) & (retL == True):
            objpoints.append(objp)
            cv2.cornerSubPix(ChessImaR, cornersR, (15, 15), (-1, -1), criteria)  # (11, 11) tablero OpenCV
            cv2.cornerSubPix(ChessImaL, cornersL, (15, 15), (-1, -1), criteria)  # Formula (size * 2 + 1)
            imgpointsR.append(cornersR)
            imgpointsL.append(cornersL)

    # Determina los nuevos valores para diferentes parámetros
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR,
                                                            ChessImaR.shape[::-1], None, None)
    hR, wR = ChessImaR.shape[:2]
    OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL,
                                                            ChessImaL.shape[::-1], None, None)
    hL, wL = ChessImaL.shape[:2]
    OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

    print('Camaras listas')

    # CALIBRAR LAS CAMARAS PARA ESTEREO
    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR,
                                                               mtxL, distL, mtxR, distR,
                                                               ChessImaR.shape[::-1],
                                                               criteria=criteria_stereo,
                                                               flags=cv2.CALIB_FIX_INTRINSIC)

    rectify_scale = 0  # 0: imagen recortada, 1: imagen no recortada
    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                      ChessImaR.shape[::-1], R, T,
                                                      rectify_scale, (0, 0))  # Ultimo parametro alfa, 0: recortado

    Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                  ChessImaR.shape[::-1], cv2.CV_16SC2)
    Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                   ChessImaR.shape[::-1], cv2.CV_16SC2)

    return Left_Stereo_Map, Right_Stereo_Map