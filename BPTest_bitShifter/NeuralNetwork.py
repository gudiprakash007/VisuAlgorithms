import json as js
from .NeuralLayer import NeuralLayer
from .Neuron import Neuron
from ...DataStructures.cpEnums import NeuralLayerType

class NeuralNetwork:

    def __init__(self):
        self._layers = [] # layers in the network
        pass
    def addLayerAt(self,posLayer,aNeuralLayer):

        numLayers = len(self._layers)
        if(numLayers==0):
            self._layers.append(aNeuralLayer)
            print('append first')
            return
        if(posLayer <= numLayers):
            self._layers.insert(posLayer,aNeuralLayer)
            print('insert middle')
        else:
            self._layers.append(aNeuralLayer)
            print('append last')

    def randomizeWeights(self):
        pass

    def updateWeights(self):
        pass

    def printNetwork(self):
        print('Number of layers:',len(self._layers))
        for aLayer in self._layers:
            #print(aLayer)
            print("---------------- Layer:",aLayer.getLayerType())
            aLayer.printLayer()

    def PrintOutput(self):
        layerCount = len(self._layers)
        oLayer = self._layers[layerCount-1]
        oLayer.PrintOutput()


    def printNetworkTrainingError(self):
        layerCount = len(self._layers)

        errVect = self._layers[layerCount-1].getLayerError()
        for oNeuron in errVect:
            pass #print("Neuron Error:",errVect[oNeuron])

        #print("Total Error:",self._layers[layerCount-1].getLayerErrorTotal())

    def RecognisePattern(self,p1,p2,p3,p4):
        # feed input to first layer ( input layer )
        inputLayer = self._layers[0]
        inputLayer.setStateNeurons([p1, p2, p3, p4])

        idx = 0;
        for aLayer in self._layers:
            #print(aLayer)
            if(0==idx):
                idx += 1
                continue # skip the input layer.
            states = aLayer.activate()
            print('**********Layer States:',states)

    def RecognisePattern(self, iPattern):
        # feed input to first layer ( input layer )
        inputLayer = self._layers[0]
        inputLayer.setStateNeurons(iPattern)

        self.ComputeForwardActivations()

        # idx = 0;
        # for aLayer in self._layers:
        #     # print(aLayer)
        #     if (0 == idx):
        #         idx += 1
        #         continue  # skip the input layer.
        #     states = aLayer.activate()
        #     #print('======> Recognition:', states)
        print("+++++  Recognize:",iPattern)
        self.PrintOutput()

    def RandomizeWeights(self):
        idx = 0
        for aLayer in self._layers:
            #print(aLayer)
            if(0==idx):
                idx += 1
                continue # skip the input layer.
            lWeights = aLayer.randomizeWeights()
            print('**********Layer Weights:',lWeights)

    def ComputeForwardActivations(self):
        layerCount = len(self._layers)
        netF = 0.0
        logF = 0.0
        for i in range(layerCount):
            #print("layer:",i," layer Type:",self._layers[i].getLayerType())
            self._layers[i].ComputeActivationValues()

    def ComputeTrainingError(self):
        layerCount = len(self._layers)
        self._layers[layerCount-1].computeTrainingActivationError()

    def ComputeNewWeights(self):
        layerCount = len(self._layers)
        netF = 0.0
        logF = 0.0
        for i in reversed(range(layerCount)):
            self._layers[i].computeTrainingNewWeights(self)

    def AdjustNewWeights(self):
        layerCount = len(self._layers)
        netF = 0.0
        logF = 0.0
        for i in reversed(range(layerCount)):
            self._layers[i].AdjustNewWeights()

    def TrainPatternBatch(self,inOutPairs):

        iter = 1;
        totErrReq = 0.05
        totErr = 1.0
        outputLayer = None
        while iter < 500 :#totErr > totErrReq:
            for ioPattern in inOutPairs:
                inputLayer = self._layers[0]  # input layer
                inputLayer.setInputTrainingPattern(ioPattern['i'])

                outputLayer = self._layers[len(self._layers) - 1]  # output layer
                outputLayer.setOutputTrainingPattern(ioPattern['o'])

                self.ComputeForwardActivations()
                self.ComputeTrainingError()
                self.ComputeNewWeights()

                self.AdjustNewWeights()
            totErr = outputLayer.getLayerErrorTotal()
            # print("========= TOtal Error:",totErr)
            # print("========= Iteration:",iter)
            totErr -= 0.2
            iter += 1

        # layerCount = len(self._layers)
        # for i in reversed(range(layerCount)):
        #     print("layer:",i," layer Type:",self._layers[i].getLayerType())

        self.printNetwork()
        #self.printNetworkTrainingError()


    def TrainPattern(self,inputPattern,outputPattern):
        inputLayer = self._layers[0] # input layer
        inputLayer.setInputTrainingPattern(inputPattern)

        outputLayer = self._layers[len(self._layers)-1] # output layer
        outputLayer.setOutputTrainingPattern(outputPattern)

        self.printNetwork()

        iter = 1;
        totErrReq = 0.05
        totErr = 1.0
        while iter < 10000 :#totErr > totErrReq:
            self.ComputeForwardActivations()
            self.ComputeTrainingError()
            self.ComputeNewWeights()

            #print("========= Iteration:",iter)
            totErr = outputLayer.getLayerErrorTotal()
            #print("========= TOtal Error:",totErr)
            totErr -= 0.2

            self.AdjustNewWeights()
            iter += 1

        # layerCount = len(self._layers)
        # for i in reversed(range(layerCount)):
        #     print("layer:",i," layer Type:",self._layers[i].getLayerType())

        self.printNetwork()
        #self.printNetworkTrainingError()

        pass

    def GetJson(self):
        print('************************************')
        layerCount = len(self._layers)
        print('Number of layers:',layerCount)
        jsonData = '\n'
        jsonData += "{ \"layers\" : [\n"
        for i in range(layerCount):
        #for aLayer in self._layers:
            aLayer = self._layers[i]
            jsonData += "{ \n \"layer\":"+str(i)+",\n"
            jsonData += "  \n \"bias\":"+str(aLayer.getLayerBias())+",\n"
            jsonData += " \"neurons\":\n"
            jsonData += " [\n"
            jsonData += aLayer.GetJson()
            jsonData += "]\n"
            jsonData += " },"
        print('************************************')
        jsonData = jsonData[:-1] + "\n"
        jsonData += " ]"
        jsonData += " }"
        return jsonData
        pass

    def LoadModel(self,jsonInFile):

        pass

    # Returns a NeuralNetwork model instance built from jsonModelFile.
    @staticmethod
    def BuildModelFromJsonFile(jsonModelFile):
        # let's try savine the NN model to a file

        fileName = jsonModelFile

        with open(fileName, "r") as inFile:
            nnModel = js.load(inFile)
            # print("json:======:",nnModel)

        nrnId = 0;
        nrnDict = {}
        aNeuralNet = NeuralNetwork()

        for layer in nnModel["layers"]:
            print(layer['layer'])
            nl = NeuralLayer()
            # print(layer['neurons'])
            for neuron in layer['neurons']:
                # print(neuron['wts'])
                numWts = len(neuron['wts'])
                inConns = {}
                for inConn in neuron['wts']:
                    # print("wts:",inConn)
                    for kvPair in inConn:
                        print("in,wt:", str(kvPair), inConn[kvPair])
                        inConns[nrnDict[str(kvPair)]] = inConn[kvPair]
                if (0 == numWts):
                    inConns = None
                nrn = Neuron(nrnId, str(neuron['nrn']), inConns)
                # print("nrn and conn:",str(neuron['nrn']), inConns)
                # print("neuron:",nrn)
                # nrn.printNeuron()
                nrnDict[neuron['nrn']] = nrn
                nrnId += 1
                nl.addNeuron(nrn, layer['bias'])
            # nl.printLayer()
            lt = NeuralLayerType.hidden

            # nl.setLayerType(NeuralLayerType.)
            aNeuralNet.addLayerAt(layer['layer'] + 1, nl)

        return aNeuralNet

