from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import os

class Polynomial_Regressor:

    def __init__(self, order):
        self.order = order
        self.weightInitialized = False
        self.modelName = ""

    def name(self, name):
        self.modelName = name

    def _getFeatureVector(self,data = []):
        data = [data]
        poly = PolynomialFeatures(self.order)
        features = poly.fit_transform(data)
        return features[0]

    def _convertToFeatureMatrix(self,x):
        matrix = []
        for i in x:
            matrix.append(self._getFeatureVector(i))
        return matrix
    
    
    def train(self, trainingData=(),validationData=(),epochs=0,batchSize=1, stepSize=0.01):
        xTrain =  np.array(self._convertToFeatureMatrix(trainingData[0]))
        xVal = np.array(self._convertToFeatureMatrix(validationData[0]))
        yTrain = np.array(trainingData[1])
        yVal = np.array(validationData[1])
        self.batchSize = batchSize
        self.features = len(xTrain[0])
        if self.weightInitialized == False:
            self.w = np.random.rand(self.features)*0.01
            self.weightInitialized = True
        for i in range(epochs):
            for j in range(0,len(xTrain),batchSize):
                batchX = xTrain[j:j+batchSize]
                batchY = yTrain[j:j+batchSize]
                self._epoch(batchX,batchY, stepSize)
            loss = self._cost(xTrain,yTrain)
            valLoss = self._valLoss(xVal,yVal)
            print("Epoch " + str(i+1) + ": | loss = " + str(self._round(loss,3)) + " validation loss = " + str(self._round(valLoss,3)))
        
    
    def _epoch(self,x,y, stepSize):
        g = self._grad(x,y)
        self.w = self.w - stepSize * g
        pass
    
    def _cost(self, x, y):
        p = self._predict(x)
        cost = 0
        #MSE:
        for i in range(len(p)):
            cost += (p[i] - y[i])**2
        cost = cost/len(p)
        return cost
    
    def _grad(self, x, y):
        p = self._predict(x)
        err = p-y
        grad = np.dot(x.T, (err / len(y)))
        return grad
    
    def _valLoss(self, x,y):
        return self._cost(x,y)

    def _predict(self, x):
        return np.dot(x,self.w)
    
    def _round(self, num, sig_figs):
        if num == 0:
            return 0
        else:
            return round(num, sig_figs - int(np.floor(np.log10(abs(num)))) - 1)

    def save(self, path=None):
        if self.modelName != "":
            if path is None:
                path = os.getcwd()
            if not os.path.isdir(path):
                print(f"The directory {path} does not exist.")
                return

            file_path = os.path.join(path, self.modelName + ".prf")

            w_str = str(self.w)
            try:
                with open(file_path, 'w') as f:
                    f.write(w_str)
            except IOError as e:
                print(f"An error occurred: {e}")
        else:
            print()
            raise Exception("Save Failed: Model name required in order to export weights")
    

        

    
