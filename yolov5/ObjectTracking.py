from scipy.spatial import distance
import numpy as np
from math import *

def calcSimilarity(obj1, obj2) -> float:
    dimension = 4

    BoundingCenter_obj1 = obj1.getBoundingCenterPos()
    distance_obj1 = obj1.getDistance()
    height_obj1 = obj1.boundingBoxHeight
    width_obj1 = obj1.boundingBoxWidth
    boundingArea_obj1 = height_obj1 * width_obj1
    vec_obj1 = np.array([BoundingCenter_obj1[0], BoundingCenter_obj1[1], distance_obj1, boundingArea_obj1])

    BoundingCenter_obj2 = obj2.getBoundingCenterPos()
    distance_obj2 = obj2.getDistance()
    height_obj2 = obj2.boundingBoxHeight
    width_obj2 = obj2.boundingBoxWidth
    boundingArea_obj2 = height_obj2 * width_obj2
    vec_obj2 = np.array([BoundingCenter_obj2[0], BoundingCenter_obj2[1], distance_obj2, boundingArea_obj2])

    covMat = np.identity(dimension)

    mah_distance = distance.mahalanobis(vec_obj1, vec_obj2, covMat)/1000
    temp1 = sqrt(abs(mah_distance))
    temp2 = pow(temp1, 2)
    temp3 = 1/(temp2 + 0.0000001)
    temp4 = exp(-pow(mah_distance, 2)*0.5)
    return temp4 * temp3


class Object:
    def __init__(self, param_leftUpper_x, param_leftUpper_y, param_rightLower_x, param_rightLower_y, param_dis):
        self.ID = None
        self.Type = None
        self.boundingBoxPos = [param_leftUpper_x, param_leftUpper_y, param_rightLower_x, param_rightLower_y]
        self.boudingCenterPos = [(self.boundingBoxPos[0] + self.boundingBoxPos[2])/2 ,
                                 (self.boundingBoxPos[1] + self.boundingBoxPos[3])/2]
        self.boundingBoxHeight = abs(self.boundingBoxPos[3] - self.boundingBoxPos[1])
        self.boundingBoxWidth = abs(self.boundingBoxPos[2] - self.boundingBoxPos[0])
        self.distance2Ego = param_dis

    def getBoundingCenterPos(self) -> list:
        return self.boudingCenterPos

    def getBoundingBoxHeight(self) -> int:
        self.boundingBoxHeight

    def getBoundingBoxWidth(self) -> int:
        self.boundingBoxWidth

    def getDistance(self) -> float:
        return self.distance2Ego

    def getID(self) -> int:
        return self.ID

    def geType(self) -> int:
        return self.Type

    def setDistance(self, params_dis):
        self.distance2Ego = params_dis

    def setBoundingBox(self, param_leftUpper_x, param_leftUpper_y, param_rightLower_x, param_rightLower_y):
        self.boundingBoxPos = [param_leftUpper_x, param_leftUpper_y, param_rightLower_x, param_rightLower_y]
        self.boudingCenterPos = [(self.boundingBoxPos[0] + self.boundingBoxPos[2]) / 2,
                                 (self.boundingBoxPos[1] + self.boundingBoxPos[3]) / 2]
        self.boundingBoxHeight = abs(self.boundingBoxPos[3] - self.boundingBoxPos[1])
        self.boundingBoxWidth = abs(self.boundingBoxPos[2] - self.boundingBoxPos[0])

    def setID(self, param_id):
        self.ID = param_id

    def setType(self, param_type):
        self.Type = param_type


class objectsFactory:
    posOccupied = 500
    Param_ObjectsNumber = 50
    Param_Thresh_similarity = 0.25

    def __init__(self):
        self.objectsPool = [None] * objectsFactory.Param_ObjectsNumber
        self.objectsPoolLastCycle = [None] * objectsFactory.Param_ObjectsNumber
        self.idPool = [None] * objectsFactory.Param_ObjectsNumber
        self.pointer2EmptyPos = 0
        self.isEmpty = True

    def createObject(self, param_tensor, param_dis) -> Object:
        objToAdd = Object(int(param_tensor[0]), int(param_tensor[1]), int(param_tensor[2]), int(param_tensor[3]), param_dis)
        while 1:
            if self.pointer2EmptyPos < objectsFactory.Param_ObjectsNumber:
                if self.idPool[self.pointer2EmptyPos] != objectsFactory.posOccupied:
                    objToAdd.setID(self.pointer2EmptyPos)
                    self.objectsPool[self.pointer2EmptyPos] = objToAdd
                    self.idPool[self.pointer2EmptyPos] = objectsFactory.posOccupied
                    self.pointer2EmptyPos += 1
                    # assert id(self.objectsPool[self.pointer2EmptyPos]) == id(objToAdd)
                    return objToAdd
                else:
                    self.pointer2EmptyPos += 1
                    #self.pointer2EmptyPos = 0 if self.pointer2EmptyPos >= objectsFactory.Param_ObjectsNumber else self.pointer2EmptyPos
            else:
                self.pointer2EmptyPos = 0

    def AssociationAndUpdate(self, param_tensor, param_dis, is_first_cycle) -> Object:
        if is_first_cycle:
            self.isEmpty = False
            retObj = self.createObject(param_tensor, param_dis)
            return retObj
        else:
            ObjToCompare = Object(int(param_tensor[0]), int(param_tensor[1]), int(param_tensor[2]), int(param_tensor[3]), param_dis)
            minimumSimilarity = 1000000
            id4minSimilarity = 0
            for Idx, currObj in enumerate(self.objectsPoolLastCycle):
                if currObj is not None:
                   similarity = calcSimilarity(currObj, ObjToCompare)
                   if similarity <= minimumSimilarity:
                       minimumSimilarity = similarity
                       id4minSimilarity = Idx
            if minimumSimilarity >= objectsFactory.Param_Thresh_similarity:
            # a existing object is matched to the measured obj, so do the updating
                #self.update(id4maxSimilarity, ObjToCompare)
                obj2Update = self.objectsPoolLastCycle[id4minSimilarity]
                obj2Update.setDistance(ObjToCompare.getDistance())
                obj2Update.setBoundingBox(int(param_tensor[0]), int(param_tensor[1]), int(param_tensor[2]), int(param_tensor[3]))
                return obj2Update
            else:
            # no existing object is matched to the measured obj, so add new obj to the objectPool
                retObj = self.createObject(param_tensor, param_dis)
                return retObj

    def update(self, Id_curr, MeasureObj) -> None:
        pass
        # obj2Update = self.objectsPool[Id_curr]
        # obj2Update.setDistance(MeasureObj.getDistance)
        # obj2Update.setBoundingBox(MeasureObj)







