import cv2
import winsound

def find_moments(frame, mask):
    gray_frame = cv2.cvtColor(mask.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_frame, 127, 255, 0)

    M = cv2.moments(thresh)
    if M["m00"] == 0:
        print("Centroide no calculable")
        center = (0, 0)
        #winsound.Beep(1000, 200)
        cv2.putText(frame, "OBJETO PERDIDO", (160, 235),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 2)
    else:
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    cv2.circle(frame, center, 5, (0, 0, 0), -1)

    return center