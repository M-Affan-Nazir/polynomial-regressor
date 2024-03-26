class Polynomial_Regressor:

    def __init__(self, order, features):
        self.order = order
        self.features = features

    def getFeatureVector(self,data = []):
        featureVector = [1]
        p = [0]

        for i in range(0,self.features):
            for j in range(1,self.order+1):
                # featureVector.append(data[i]^j)
                featureVector.append(str(data[i]) + "**" + str(j))
                p.append(j)
        
        
        for i in range(1,(self.order*self.features)):
            # print("i: " + str(i))
            start = i+(self.order+1)-p[i]
            # print("start : " + str(start))
            # print("p : " + str(p[i]))
            for j in range(start,self.order*self.features+1):
                # print(j)
                if( (p[i]+p[j]) <= self.order ):
                    # featureVector.append( data[i]*data[j] )
                    featureVector.append(str(featureVector[i]) + " * " + str(featureVector[j]))
                # else:
                    # print("OB : " + str(featureVector[i]) + "*" + str(featureVector[j]))
            # print()
            pass


        for counter in range(1, 2**len(data)):
            combination = []
            for j in range(len(data)):
                if counter & (1 << j):
                    combination.append(data[j])
            if len(combination) > 2:
                featureVector.append(''.join(combination))

        for i in featureVector:
            print(i)
            
            
            
        
        

    
