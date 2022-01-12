import numpy as np

global drawnItems, focusedJoint

def rot(theta):
    #Function performs a 2-D matrix rotation based on theta
    R = np.matrix([[np.cos(theta), (-1)*np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return R

def eye(val):
    #function creates val x val identity matrix
    ident = []
    for i in range(0,val):
        tempI = [0]*val
        tempI[i] = 1
        ident.append(tempI)
    identM = np.asarray(ident)
    #print(identM)
    return identM

def pixCoor(x,y,width,height):
    u = x 
    v = y
    return [u,v]
    
class RobotArm:
    global Q, P, p0T, r0T, sgmentList, numSegs, drawnItems, focusedJoint
    sgmentList = [] #List of segments
    numSegs = 0 #The number of segments this robot contains
    p0T = np.matrix([[0],[0]]) #The vector from the origin to the end effector
    r0T = eye(2) #The rotation from the origin to the end effector
    Q = [] # List of joint angles
    P = [] #List of each joint position
    focusedJoint = 0
    drawnItems = []
    def __init__(self, segments, zeroConfig):
        global Q, P, p0T, r0T, sgmentList, numSegs, drawnItems, focusedJoint, ax
        focusedJoint = 0
        drawnItems = []
        P.append(p0T)
        for i in range(0,len(segments)):
            r0T = np.dot(r0T,rot(zeroConfig[i]))            
            p0T = p0T + np.dot(r0T,segments[i].getLength())
            (Q).append(zeroConfig[i])
            (P).append(p0T)
        numSegs = len(segments)
        sgmentList = segments
     
    #def setJointAngle(self,joint,angle):
    def modJointAngle(joint, mod):
        global Q, P, p0T, r0T, sgmentList, numSegs
        Q[joint] = Q[joint] + mod
        print ("QQ",Q)
        p0T = np.matrix([[0],[0]]) #The vector from the origin to the end effector
        r0T = eye(2) #The rotation from the origin to the end effector
        P = [] #List of each joint position
        
        P.append(p0T)
        for i in range(0,numSegs):
            r0T = np.dot(r0T,rot(Q[i]))            
            p0T = p0T + np.dot(r0T,sgmentList[i].getLength())
            (P).append(p0T)
    
    def getJointPositionsxy():
        global Q, P
        positions = []
        for i in P:
            templist = []
            templist.append(float(i[0]))
            templist.append(float(i[1]))
            positions.append(templist)
        return positions   
