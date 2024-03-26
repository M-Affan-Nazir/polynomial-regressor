from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class Polynomial_Regressor:

    def __init__(self, order, features):
        self.order = order
        self.features = features

    def getFeatureVector(self,data = []):
        data = [data]
        poly = PolynomialFeatures(self.order)
        features = poly.fit_transform(data)
        print(len(features[0]))
            
            
            
        
        

    
