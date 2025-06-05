import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CostFunction:
    def __init__(self):
        self.errors = {
            "mse":[self.MSE,self.MSE_Derivative] #mean square error
            #can add more error functions like this
        }

    def MSE(self,predicted_output,output):
        return np.mean((predicted_output-output)**2)

    def MSE_Derivative(self,input,predicted_output,output):
        cost = predicted_output - output
        # (bias slope,weight slope)
        return 2*np.mean(cost),2*np.mean(cost*input)


class LinearRegression:
    def __init__(self,input,output,learning_rate = 0.001):
        self.input = input
        self.output = output
        self.learning_rate = learning_rate
        n = self.input.shape[1]
        self.weights = np.random.uniform(0,n)
        self.bias = np.random.uniform(0,1)

    def Forward_Propagation(self):
        return np.multiply(self.input,self.weights) + self.bias

    def Backward_Propagation(self,cost_function):
        cost_method = CostFunction().errors[cost_function]
        predicted_output = self.Forward_Propagation()
        cost = cost_method[0](predicted_output,self.output)
        db,dw = cost_method[1](self.input,predicted_output,self.output)

        #updating
        self.weights -= dw*self.learning_rate
        self.bias -= db*self.learning_rate

        return cost

    def train(self,cost_function,epochs):
        cost_function = cost_function.lower()
        errors = []
        for i in range(epochs):
            cost = self.Backward_Propagation(cost_function)
            errors.append(cost)
            print("EPOCH : " , i+1 , " " + cost_function.upper() + " " ,cost)

        plt.plot(list(range(epochs)),errors)
        plt.xlabel("Epochs")
        plt.ylabel("Error")

        plt.show()


data = pd.read_csv("data.csv")

data = data.dropna()

input = np.array(data.x[0:500]).reshape(500, 1)
output = np.array(data.y[0:500]).reshape(500, 1)

#if learning rate is too high then error may increase or may oscillate
#if learning rate is too low then it make take a very a lot of iterations
r = LinearRegression(input,output,0.0001)
r.train("mse",100)



