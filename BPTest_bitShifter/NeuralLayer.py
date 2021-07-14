from ...DataStructures.cpEnums import *
#https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

class NeuralLayer:

    def __init__(self):
        self._neurons = [] # set of neurons in the layer
        self._layerError = {} #None # (key,value) = (neuron,errVal)
        self._totalError = 0.0
        self._layerType = NeuralLayerType.hidden
        self._backError = None
        self._bias = 1.0
        #self._biasWts = {} # key,value = neuron,wt
        pass

    def addNeuron(self,aNeuron,biasWt):
        aNeuron.setBiasWt(biasWt)
        self._neurons.append(aNeuron)
        #self._biasWts[aNeuron] = biasWt
        #print("adding neuron:",aNeuron._id)

    def updateWeights(self):
        pass

    def printLayer(self):
        print('Number of neurons:',len(self._neurons),'\n')
        for aNeuron in self._neurons:
            aNeuron.printNeuron()
            print('----\n')
        print('--------------------------------------------------------------\n')

    def PrintOutput(self):
        oStr = ''
        for oNeuron in self._neurons:
            oStr += oNeuron.getLabel()+":"+str(oNeuron.getState())+"  "
        print(oStr)

    def activate(self):
        #print('Layer State:')
        retState = ''
        for aNeuron in self._neurons:
            # act based on input from connected lower lavel neurons
            aNeuron.activate(self._bias)
            #print("activated State:",aNeuron.getState())
            retState += str(aNeuron.getState())
        return retState

    def ComputeActivationValues(self):
        return self.activate()

    def randomizeWeights(self):
        print('Layer Weights:')
        retWt = ''
        for aNeuron in self._neurons:
            # act based on input from connected lower lavel neurons
            retWt += str(aNeuron.randomizeWeights())
        return retWt

    def AdjustNewWeights(self):
        if(NeuralLayerType.input==self._layerType):
            return # exclude Input layer from weight adjustment.
        for aNeuron in self._neurons:
            aNeuron.adjustNewWeights()

    def setStateNeurons(self, iv):
        idx=0
        lIv = len(iv)
        for v in iv:
            if(idx<lIv):
                (self._neurons[idx]).setState(v)
                idx += 1

    def setInputTrainingPattern(self, iv):
        self._layerType = NeuralLayerType.input
        self.setStateNeurons(iv)

    def setOutputTrainingPattern(self, ov):
        self._layerType = NeuralLayerType.output
        idx=0
        lIv = len(ov)
        #self._layerError = [0.0]*lIv
        for v in ov:
            if(idx<lIv):
                (self._neurons[idx]).setExpectedState(v)
                idx += 1

    def computeTrainingActivationError(self):
        layerCount = len(self._layerError)
        retWt = ''
        self._totalError = 0.0 #
        for aNeuron in self._neurons:
            theDiff = aNeuron.getExpectedState() - aNeuron.getState()
            self._layerError[aNeuron] = theDiff*theDiff/2.0
            self._totalError += self._layerError[aNeuron]
        # for i in range(layerCount):
        #     self._layerError[i] = self._layerError[i].getExpectedState() - self._layerError[i].getState()
        #     self._totalError += self._layerError[i]


    def computeTrainingNewWeights(self,allLayers):
        idxNeuron=0
        retWt = ''

        # rate of change of total error at Output unit wrt an input neuron Weight is given by
        # = neuron's activation error * actual state * ( 1- actual state ) * input(indrite) actual State
        # let's call it TotalErrorChangePerWeight
        # wtDelta = learning rate * TotalErrorChangePerWeight
        # New weight = OldWeight + wtDelta

        if(NeuralLayerType.output == self._layerType):
            for aNeuron in self._neurons:
                nrnActErr = self._layerError[aNeuron]
                aNeuron.computeNewWeights(nrnActErr)
            return

        if(NeuralLayerType.hidden == self._layerType):
            for aNeuron in self._neurons:
                dOutBydNet = aNeuron.getState()*(1-aNeuron.getState())
                iNeuronTrainWts = aNeuron.getInputNeuronsTrain()
                outNeurons = aNeuron.getAxons()
                #print("Neuron:",aNeuron.getLabel())
                pSum = 0.0
                for oNrn in outNeurons:
                    #print("o:",oNrn.getLabel())
                    iNrns = oNrn.getInputNeurons()
                    #print("expected-actual:",oNrn.getState()-oNrn.getExpectedState())
                    #print("dOutBydNet,wt:",oNrn.getState()*(1.0-oNrn.getState()),",",iNrns[aNeuron])
                    pSum += (oNrn.getState()-oNrn.getExpectedState())*oNrn.getState()*(1.0-oNrn.getState())*iNrns[aNeuron]
                    #print("pSum:",pSum)
                for iNeuron in iNeuronTrainWts:
                    #print("  In Neuron:",iNeuron.getLabel())
                    dNetBydWt = iNeuron.getState()
                    #print('    dOutBydNet,dNetBydWt:(',dOutBydNet,',',dNetBydWt,')')
                    iNeuronTrainWts[iNeuron] = iNeuronTrainWts[iNeuron] - 0.5*dOutBydNet*dNetBydWt*pSum
                    #print("new weight:",iNeuronTrainWts[iNeuron])

            return



    def getLayerError(self):
        return self._layerError

    def getLayerErrorTotal(self):
        return self._totalError

    def getLayerType(self):
        return self._layerType

    def setLayerType(self,lt):
        self._layerType = lt

    def getLayerBias(self):
        return self._bias

    def setLayerBias(self, lBias):
        self._bias = lBias

    def GetJson(self):
        jData = ""
        for aNeuron in self._neurons:
            jData += aNeuron.GetJson() + ","
        jData = jData[:-1]
        return jData #"name:{Prakash}"