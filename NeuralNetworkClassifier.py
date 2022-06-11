import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetworkClassifier():
    def __init__(self, sampleDem, hiddenLayerNum=5, learningRate=0.1):
        '''
        Create a instance, the input parameters are the dimensions of the sample and the number of hidden layers, which are used to control the network structure, 
        the default number of neurons in hidden layers is 5, and the learning rate is 0.1 by default
        All parameters are initialized by generating random numbers
        The number of neurons in the output layer is 1
        '''
        self.hiddenLayerNum = hiddenLayerNum
        self.sampleDem = sampleDem
        self.learningRate = learningRate
        #The initial values of the weights from the input layer to the hidden layer are random numbers between (0,1)
        self.firstProcess = np.mat(np.random.random(size=(sampleDem, hiddenLayerNum)))
        #The initial values of the weights from the hidden layer to the output layer are random numbers between (0,1)
        self.secondProcess = np.mat(np.random.random(size=(hiddenLayerNum, 1)))
        self.hiddenLayerThreshold = np.mat(np.random.random(size=(1, hiddenLayerNum)))#Initialize the hidden layer threshold
        self.outputLayerThreshold = np.random.random()#Initialize the output layer threshold
        
    @staticmethod
    def Sigmoid(x):
        '''The sigmoid function can be used as the activation function for the hidden and output layers'''
        return 1/(1+np.exp(-x))
    
    @staticmethod
    def ReLU(x):
        '''
        The hidden layer activation function can also use ReLU
        '''
        return 0 if x < 0 else x
      
    def outputLayerGradient(self, forecast, target):
        '''
        Generate the gradient of the output layer, since the number of neurons in the output layer is 1, the return value should be an m*1 matrix
        '''
        outputGradient = np.zeros((sampleNum, 1))
        for i in range(sampleNum):
            outputGradient[i, 0] = forecast[i]*(1-forecast[i])*(target[i]-forecast[i])
        return outputGradient
    
    def hiddenLayerGradient(self, outputGradient, hiddenLayerOutput):
        '''
        Generate the gradient of the hidden layer and return a n*1 matrix, where n is the number of sample dimensions
        '''
        hiddenGradient = np.zeros((sampleNum, self.hiddenLayerNum))
        for i in range(sampleNum):
            for j in range(self.hiddenLayerNum):
                hiddenGradient[i, j] = hiddenLayerOutput[i, j]*(1-hiddenLayerOutput[i, j])*self.secondProcess[j, 0]*outputGradient[i, 0]
        return hiddenGradient  
        
      
    def forecast(self, inputData, activation='Sigmoid'):
        '''
        The method generates the output based on the input samples according to the current network structure parameters, 
        if the number of samples is m, the final output should be an m*1 matrix
        The activation function currently uses Sigmoid and ReLU, where the default is Sigmoid.
        Return network output and hidden layer output
        '''
        sampleNum, sampleDem = np.shape(inputData)#Number and dimension of samples
        if sampleDem != self.sampleDem:
            print('The dimensionality of the sample input is not consistent with the model')
            SystemExit()
        hiddenLayerInput = np.matmul(inputData, self.firstProcess)#The input of the hidden layer, which should be a matrix of m*hiddenLayerNum
        #the output of the hidden layer should be consistent with the dimension of the input
        hiddenLayerOutput = np.mat(np.zeros((sampleNum, self.hiddenLayerNum)))
        for i in range(sampleNum):
            for j in range(self.hiddenLayerNum):
                if activation == 'Sigmoid':
                    hiddenLayerOutput[i, j] = self.Sigmoid(hiddenLayerInput[i, j] - self.hiddenLayerThreshold[0, j])# output of the hidden layer
                elif activation == 'ReLU':
                    hiddenLayerOutput[i, j] = self.ReLU(hiddenLayerInput[i, j] - self.hiddenLayerThreshold[0, j])# output of the hidden layer
                else:
                    print('The activation function is not defined yet')
        outputLayerInput = np.matmul(hiddenLayerOutput, self.secondProcess)#The input of the output layer, which should be an m*1 matrix
        #output value
        output = np.zeros((sampleNum, 1))
        for i in range(sampleNum):
            output[i] = self.Sigmoid(outputLayerInput[i, 0] - self.outputLayerThreshold)
        return output, hiddenLayerOutput
    
    @staticmethod
    def error(forecast, target):
        '''
        Calculate the error between the forecast and true results, 
        return the error for each sample and the cumulative error 
        '''
        error = 0.5 * np.array(forecast - target)**2#error for each sample
        cumulativeError = np.average(error)#cumulative error
        return error, cumulativeError
        
    
    def training(self, inputData, target, trainingMethod='CGD', activation='Sigmoid',maxIteration=500, fitResult=False):
        '''
        inputData should be a m*n np.mat matrix, containing m samples, each sample is an n-dimensional vector
        target is m*1 matrix, i.e., each sample output
         'CGD' is cumulative gradient descent, 'GD' is standard gradient descent (adjust the model parameters one by one according to the sample order), 
         trainingMethod is cumulative gradient descent algorithm by default
        maxIteration is the maximum number of iterations, default is 500
        fitResult parameter is used to control whether to output the cumulative error and correct classification rate of each iteration as well as the error curve
        ,the default is False
        '''
        global sampleNum#globalize the number of samples
        sampleNum = np.shape(inputData)[0]
        time = 0#used to control number of iterations
        sampleIndex = 0#select the sample to train in standard gradient descent
        result = []#Store the error of each iteration
        while (time < maxIteration):
            output, hiddenLayerOutput = self.forecast(inputData, activation=activation)#output of the forecast result
            
            if fitResult == True:
                error,cumulativeError = self.error(output, target)#Calculate the mean sample error and cumulative error
                result.append(cumulativeError)
                #Prediction accuracy
                print('The cumulative error is %g，%d/%d are correctly classfied'%(cumulativeError, np.sum(abs(output-target)<0.5), sampleNum))
        
            outputGradient = self.outputLayerGradient(output, target)#the gradient of the output layer
            hiddenGradient = self.hiddenLayerGradient(outputGradient, hiddenLayerOutput)#the gradient of the hidden layer
            #update parameters of the neural network
            if trainingMethod == 'CGD':
                self.outputLayerThreshold -= self.learningRate*np.average(outputGradient, axis=0)#update threshold of the output layer
                self.secondProcess += self.learningRate*np.matmul(hiddenLayerOutput.transpose(), outputGradient)/sampleNum
                self.hiddenLayerThreshold -= self.learningRate*np.average(hiddenGradient, axis=0)#update threshold of the hidden layer
                self.firstProcess += self.learningRate*np.matmul(inputData.transpose(), hiddenGradient)/sampleNum
            elif trainingMethod == 'GD':
                if sampleIndex == (sampleNum - 1):
                    sampleIndex = 0
                else:
                    sampleIndex += 1
                self.outputLayerThreshold -= self.learningRate*outputGradient[sampleIndex,0]#update threshold of the output layer
                self.secondProcess += self.learningRate*np.matmul(hiddenLayerOutput.transpose()[:, sampleIndex], np.mat(outputGradient[sampleIndex,0]))
                self.hiddenLayerThreshold -= self.learningRate*hiddenGradient[sampleIndex, :]#update threshold of the hidden layer
                self.firstProcess += self.learningRate*np.matmul(inputData.transpose()[:, sampleIndex], np.mat(hiddenGradient[sampleIndex, :]))
            else:
                print('no such training method yet')
                system.exit()
            time += 1
        if fitResult == True:
            plt.plot(result)
            plt.show()

    
    
if __name__ == '__main__':
    #test the network
    inputData = np.mat(
        [[1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 0.697, 0.46 ],
        [0.5  , 1.   , 0.5  , 1.   , 1.   , 1.   , 0.774, 0.376],
        [0.5  , 1.   , 1.   , 1.   , 1.   , 1.   , 0.634, 0.264],
        [1.   , 1.   , 0.5  , 1.   , 1.   , 1.   , 0.608, 0.318],
        [0.   , 1.   , 1.   , 1.   , 1.   , 1.   , 0.556, 0.215],
        [1.   , 0.5  , 1.   , 1.   , 0.5  , 0.5  , 0.403, 0.237],
        [0.5  , 0.5  , 1.   , 1.   , 0.5  , 0.5  , 0.481, 0.149],
        [0.5  , 0.5  , 1.   , 1.   , 0.5  , 1.   , 0.437, 0.211],
        [0.5  , 0.5  , 0.5  , 0.5  , 0.5  , 1.   , 0.666, 0.091],
        [1.   , 0.   , 0.   , 1.   , 0.   , 0.5  , 0.243, 0.267],
        [0.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.245, 0.057],
        [0.   , 1.   , 1.   , 0.   , 0.   , 0.5  , 0.343, 0.099],
        [1.   , 0.5  , 1.   , 0.5  , 1.   , 1.   , 0.639, 0.161],
        [0.   , 0.5  , 0.5  , 0.5  , 1.   , 1.   , 0.657, 0.198],
        [0.5  , 0.5  , 1.   , 1.   , 0.5  , 0.5  , 0.36 , 0.37 ],
        [0.   , 1.   , 1.   , 0.   , 0.   , 1.   , 0.593, 0.042],
        [1.   , 1.   , 0.5  , 0.5  , 0.5  , 1.   , 0.719, 0.103]])
    
    target = np.mat([[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], 
                    [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]])
    sampleDem = np.shape(inputData)[1]
    #Build neural network model, the number of hidden layer units and learning rate are not adjusted, which is 5 and 0.1
    m = NeuralNetworkClassifier(sampleDem, learningRate=0.1)
    '''
    The following training parameters indicate that the training method uses the cumulative gradient descent algorithm, 
    the maximum number of iterations is 1500, the cumulative error and classification accuracy of each iteration are printed, 
    and the hidden layer activation function uses ReLU
    '''
    m.training(inputData, target, trainingMethod='CGD',maxIteration=1500,fitResult=True, activation='ReLU')
    #test the neural network on a new sample
    testData = np.mat([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.675, 0.284])
    #testData = np.mat([0.5, 0.5, 1, 0.5,0,1.0,0.38,0.12])
    output = m.forecast(testData, activation='ReLU')[0]
    print(output)
