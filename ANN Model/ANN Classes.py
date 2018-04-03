import random
import math
import numpy

class OutputNode(object):
    """Node for artificial neural net"""

    bias = 0.5
    numWeights = 0
    weights = [] #w
    errorTerms = 0 #For averaging error sum
    errorSums = [] #dE/dw
    biasError = 0
    output = 0
    nodeError = 0 #delta

    def __init__(self, numWeights, weights=[], bias=0.5):

        self.numWeights = numWeights
        self.bias = random.random()

        #if given weights use them
        if len(weights) != 0:
            self.weights = weights
        #otherwise assign random weights
        else:
            for lcv in range(0,numWeights):
                weights.append(random.random())
        #init error sums
        self.errorSums = [0]*self.numWeights
        self.biasError = 0

    """Calculates the output of the node"""
    def forwardCalc(self, inputs):
        self.output = numpy.dot(self.weights, inputs)
        return self.output

    """Calculates the node error from weights to next layer and error of corresponding nodes"""
    def errorCalc(self, weights, errors):
        # expected - actual
        self.nodeError = errors - self.output

    """Finishes backcalculation of error for each weight then returns node Error"""
    def backCalc(self, weights, errors, inputs):
        #calculate node error
        self.errorCalc(weights, errors)
        #add to each error sum
        for lcv in range(0,self.numWeights):
            self.errorSums[lcv] = self.errorSums[lcv] + self.nodeError*inputs[lcv]
        #add to bias error
        self.biasError += self.nodeError*self.bias
        #increment error count
        self.errorTerms += 1
        return self.nodeError

    """Adjusts weights"""
    def adjstWeight(self, learnRate):
        #adjust weights
        for lcv in range(0, self.numWeights):
            self.weights[lcv] -= learnRate*(self.errorSums[lcv]/self.errorTerms) # -a*(dE/dw)
        #adjust bias
        self.bias -= learnRate*(self.bias/self.errorTerms)
        #clear error values
        self.errorTerms = 0
        #clear error sums
        self.errorSums = [0] * self.numWeights
        #clear bias error
        self.biasError = 0

    """Returns weights to be used for backcalc"""
    def getWeights(self):
        return self.weights

class HiddenNode(OutputNode):

    """sigmoid function output for value"""
    def sigmoid(self, value):
        return 1/(1+math.exp(value))

    """adds sigmoid to forward calculation"""
    def forwardCalc(self, inputs):
        #sigmoid of super output
        self.output = self.sigmoid(super(HiddenNode, self).forwardCalc(inputs))
        return self.output

    """calculates error for hidden node"""
    def errorCalc(self, weights, errors):
        self.nodeError = self.output*(1 - self.output)*numpy.dot(weights, errors) #g'(a)*Sum(w*e)

class InputNode(OutputNode):

    """sets bias to value for it to be returned as the value"""
    def __init__(self, value):
        super(InputNode, self).__init__(0,bias=value)

    """"""
    def forwardCalc(self, inputs):
        return self.bias

    """calculates error for hidden node"""
    def errorCalc(self, weights, errors):
        self.nodeError = self.output * (1 - self.output) * numpy.dot(weights, errors)  # g'(a)*Sum(w*e)

    """Doesn't due full backcalc only returns error of node"""
    def backCalc(self, weights, errors, inputs):
        self.errorCalc(weights, errors)
        return self.nodeError

class Layer(object):
    numNodes = 0
    nodes = []
    outputs = []
    weights = [[]]
    errors = []

    def __init__(self, numNodes = 0, nodeType=OutputNode, nodes=[], inputsSize=0):
        # check if nodes are given
        if len(nodes) != 0:
            self.numNodes = len(nodes)
            # create each node
            for lcv in range(0,self.numNodes):
                #TODO: add check for being of OutputNode type
                currNode = nodes[lcv]
                #node is array of [numWeights, weights[], bias]
                self.nodes.append(nodeType(currNode[0],currNode[1],currNode[2]))
                self.weights.append(currNode[1])
        #otherwise cunstruct new nodes
        else:
            self.numNodes = numNodes
            for lcv in range(0,self.numNodes):
                #append blank nodes
                self.nodes.append(nodeType(inputsSize))
                self.weights.append(self.nodes[lcv].getWeights())
        # init outputs
        outputs = [0]*self.numNodes
        # init errors
        errors = [0]*self.numNodes

    """Returns the current node outputs (updated at forward calc)"""
    def getOutputs(self):
        return self.outputs

    """forward calculate output for each node and store in outputs"""
    def forwardCalc(self, inputs):
        for lcv in range(0,self.numNodes):
            self.outputs[lcv] = self.nodes[lcv].forwardCalc(inputs)
        return self.outputs

    """Back calculates for all nodes"""
    def backCalc(self, weights, errors, inputs):
        #transpose weights to get weight for current node at each output
        weightsTranspose = numpy.array(weights).transpose().tolist()
        for lcv in range(0, self.numNodes):
            self.errors[lcv] = self.nodes[lcv].backCalc(weights[lcv], errors, inputs)
        return self.errors, self.weights

    """adjusts weights and returns current weights"""
    def adjustWeights(self, learnRate):
        for lcv in range(0, self.numNodes):
            # update node weights
            self.nodes[lcv].adjustWeights(learnRate)
            # store new weights
            self.weights[lcv] = self.nodes[lcv].getWeights()
        return

class NerualNet(object):
    inputLayer = None
    numInputNodes = 0
    hiddenLayers = []
    numHiddenLayers = 0
    outputLayer = None
    numOutputNodes = 0
    learnRate = 0

    def __init__(self, numInputNodes, numOutputNodes, numHiddenNodes, numHiddenLayers, learnRate, layers=[]):
        self.learnRate = learnRate
        #check if constructing new net
        if len(layers) != 0:
            self.numHiddenLayers = len(layers) - 1
            self.numOutputNodes = len(layers[-1])
            self.numInputNodes = numInputNodes
            # create output layer
            self.outputLayer = Layer(numOutputNodes, nodeType=OutputNode, nodes=layers[-1], inputsSize=self.numOutputNodes)
            # create first hidden layer
            self.hiddenLayers[0] = Layer(numHiddenNodes, nodeType=HiddenNode, nodes=layers[0], inputsSize=len(layers[0]))
            if numHiddenLayers > 1:
                # create all other hidden layers
                for lcv in range(1,numHiddenLayers):
                    self.hiddenLayers[lcv] = Layer(numHiddenNodes, nodeType=HiddenNode, nodes=layers[lcv], inputsSize=len(layers[lcv]))
        #new net
        else:
            self.numHiddenLayers = numHiddenLayers
            self.numOutputNodes = numOutputNodes
            self.numInputNodes = numInputNodes
            # create blank layers
            self.outputLayer = Layer(numOutputNodes, nodeType=OutputNode, inputsSize=numHiddenNodes)
            # create first hidden layer
            self.hiddenLayers[0] = Layer(numHiddenNodes, nodeType=HiddenNode, inputsSize=numInputNodes)
            if numHiddenLayers > 1:
                # create all other hidden layers
                for lcv in range(1,numHiddenLayers):
                    self.hiddenLayers[lcv] = Layer(numHiddenNodes, nodeType=HiddenNode, inputsSize=numHiddenNodes)
        #create blank input layer
        self.inputLayer = Layer(self.numInputNodes, nodeType=InputNode)

    """Sets the input layer to the new inputs"""
    def setInputLayer(self,inputs):
        inputNodes = []
        #build array of nodes
        for lcv in range(0,self.numInputNodes):
            inputNodes.append([0,[],inputs[lcv]])
        #create layer from array
        self.inputLayer = self.inputLayer = Layer(self.numInputNodes, nodeType=InputNode, nodes=inputNodes)

    """forward calculation"""
    def forwardCalc(self):
        #TODO: input layer could stay as an array
        #get inputs
        lastOutput = self.inputLayer.forwardCalc([])
        #calculate hidden layers
        for lcv in range(0,self.numHiddenLayers):
            lastOutput = self.hiddenLayers[lcv].forwardCalc(lastOutput)
        #calculate outputs and return
        return self.outputLayer.forwardCalc(lastOutput)

    """back calculation"""
    def backCalc(self, expectedOutputs):
        error = []
        #calculate output error
        for output in self.outputLayer.getOutputs():
            error.append(expectedOutputs - output)
        #backcalc on output layer
        error, weights = self.outputLayer.backCalc([],error,self.hiddenLayers[-1].getOutputs())
        #backcalc hidden layers
        if self.numHiddenLayers > 1:
            for lcv in range(self.numHiddenLayers - 1, 0, -1):
                error, weights = self.hiddenLayers[lcv].backCalc(weights, error, self.hiddenLayers[lcv-1].getOutputs)
            #backcalc last hidden layer
            error, weights = self.hiddenLayers[0].backCalc(weights, error, self.hiddenLayers[1].getOutputs)
        else:
            error, weights = self.hiddenLayers[0].backCalc(weights, error, self.outputLayer.getOutputs)
        #calc error at inputs
        error, weights = self.inputLayer.backCalc(weights,error,self.hiddenLayers[0].getOutputs())
        #return error at input
        return error

    #TODO: add to array methods

    """adjust weights"""
    def adjWeight(self):
        for layer in self.hiddenLayers:
            layer.adjustWeights(self.learnRate)

    """adjust learn rate"""
    def adjLearnRate(self, learnRate):
        self.learnRate = learnRate

    """applies learning cycle for given input output pairs"""
    def learnCycle(self,inputOutputPairs, printing=False):
        for pair in inputOutputPairs:
            input = pair[0]
            expectedOutput = pair[1]
            #set input
            self.setInputLayer(input)
            #forward calculate
            output = self.forwardCalc()
            #backward calculate
            error = self.backCalc(expectedOutput)
            if printing:
                print("\n")
                print("Inputs:  "+ input)
                print("Outputs: "+ output)
                print("Error:   "+ error)
        if printing:
            print("\n")

    """"""