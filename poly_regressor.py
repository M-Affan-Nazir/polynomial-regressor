from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import os

class Polynomial_Regressor:

    def __init__(self, order):
        self.order = order

    def _getFeatureVector(self,data = []):
        data = [data]
        poly = PolynomialFeatures(self.order)
        features = poly.fit_transform(data)
        return features[0]
    
    def setEpochs(self,epochs):
        self.epochs = epochs
    
    def train(self, features,target):
        self.features = len(features[0])
        self.w = np.random.rand(self.features)*0.01
        for i in range(self.epochs):
            for j in range(len(features)):
                print("Epoch: " + str(i) + "  |  Percentage Completed: " + str(int(j/len(features)*100)))
                self._epoch(features[j],target[j])
            print("\n")

    
    def _epoch(self,x,y):
        predicted = self._predict(x)
        cost = self._cost(x,y)
        g = np.array(x)*(np.array(predicted-y))
        self.w = self.w - ((1+abs(g))**(-1))*g
    
    def _cost(self, x, y):
        p = self._predict(x)
        #MSE:
        return (p - y)**2
        
    
    def _predict(self, x):
        return np.dot(x,self.w)

        
    
    

        

    
