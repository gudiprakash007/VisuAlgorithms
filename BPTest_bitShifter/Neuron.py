from ...Utils.UFunc import UtilFuncs as UF
from random import *

class Neuron:
    """Neuron class"""

    def __init__(self,inId,inAId,inConns):
        """
        Accepts an Identifier for the neuron and input connection neurons.

        Parameters
        ----------
        inId : int
            The identity for the neuron
        inConns : , optional
            dictionary where key is Neuron (input) instance and value is
            the connection weight between this and the input.
        Raises
        ------
        NotImplementedError

        """

        if(None == inConns):
            """input neurons: dictionary - key:Neuron, value:weight"""
            self._indritesWt = {}
            """input neurons: dictionary - key:Neuron, value:delta weight
            connection weights(delta) from  this neuron- Used in TRAINING"""
            self._indritesWtTrain = {}
        else:
            self._indritesWt = inConns
            self._indritesWtTrain = {}

        # print('number of inbound Trains:',len(self._indritesWt))
        # print('number of inbound wt Trains:',len(self._indritesWtTrain))
        trainCnt = 0;
        for aNeuron in self._indritesWt:
            self._indritesWtTrain[aNeuron] = self._indritesWt[aNeuron]# uniform(0, 0.001)
            trainCnt += 1
        # print('number of inbound Weight Trains Added:', len(self._indritesWtTrain))

        self._id = inId
        self._aid = inAId
        self._biasWt = 0.0
        self._expectedState = 0.0
        self._threashold = 0.0
        self._state = 0.0

        self._axons = [] # array of output neurons

        pass

    def setIndrites(self,inConns):
        if(None == inConns):
            self._indritesWt = {}
        else:
            self._indritesWt = inConns


    def setAxons(self,inAxons):
        self._axons = inAxons

    def getAxons(self):
        return self._axons

    def activate(self,bias):
        wSum = 0.0
        numOfInputs = len(self._indritesWt)
        if(0==numOfInputs):
            #must be input layer
            sigVal = self._state
            #print("State: ",sigVal)
            return

        for aNeuron in self._indritesWt:
            #print('input state and wt:',key.getState(),'-',self._indrites[key])
            wSum += (float(aNeuron.getState()) * self._indritesWt[aNeuron])
        #print('wSum vs threashold:',wSum,'-',self._threashold)
        wSum += self._biasWt * bias

        sigVal = UF().Sigmoid(wSum)
        print('Sigmoid:',sigVal)

        # if(sigVal >= self._threashold):
        #     self._state = 1.0
        # else:
        #     self._state = 0.0

        self._state = sigVal

        # if(wSum > self._threashold):
        #     self._state = 1
        # else:
        #     self._state = 0.0
        pass

    def randomizeWeights(self):
        wVals = ''
        for key in self._indritesWt:

            self._indritesWt[key] = uniform(0, 0.001)
            wVals += ' '
            wVal = "{:.4f}".format(self._indritesWt[key])
            wVals += str(wVal)
        return wVals

    def randomizeWeightsTrain(self):
        wVals = ''
        for key in self._indritesWtTrain:

            self._indritesWtTrain[key] = uniform(0, 0.001)
            wVals += ' '
            wVal = "{:.4f}".format(self._indritesWtTrain[key])
            wVals += str(wVal)
        return wVals

    def computeNewWeights(self,actErr):
        actState = self.getState()
        inNeuronsTrain = self.getInputNeuronsTrain()
        #print("nrn Out:",actState)
        for inNeuronT in inNeuronsTrain:
            # print("--------actErr:",actErr)
            # print("nrn eTot/dOut:",actState-self._expectedState)
            # print("dOut/dNet:",actState*(1.0-actState))
            # print("dNet/dWt:", inNeuronT.getState())
            TotErrChngPerInputNeuronWt = (actState-self._expectedState)*actState*(1.0-actState)*inNeuronT.getState()
            #print('--------TotalErrChange Per This Neuron:', TotErrChngPerInputNeuronWt)
            self._indritesWtTrain[inNeuronT]=self._indritesWtTrain[inNeuronT] - 0.5 * TotErrChngPerInputNeuronWt
            #print("new Wt:",self._indritesWtTrain[inNeuronT])

    def computeNewWeightsHidden(self,dOutBydNet,dNetBydWt):
        actState = self.getState()
        inNeuronsTrain = self.getInputNeuronsTrain()
        #print("nrn Out:",actState)
        for inNeuronT in inNeuronsTrain:
            #print("--------actErr:",actErr)
            # print("nrn eTot/dOut:",actState-self._expectedState)
            # print("dOut/dNet:",actState*(1.0-actState))
            # print("dNet/dWt:", inNeuronT.getState())
            TotErrChngPerInputNeuronWt = (actState-self._expectedState)*actState*(1.0-actState)*inNeuronT.getState()
            # print('--------TotalErrChange Per This Neuron:', TotErrChngPerInputNeuronWt)
            self._indritesWtTrain[inNeuronT]=self._indritesWtTrain[inNeuronT] - 0.5 * TotErrChngPerInputNeuronWt
            # print("new Wt:",self._indritesWtTrain[inNeuronT])

    def adjustNewWeights(self):
        inNeuronsTrain = self.getInputNeuronsTrain()
        for inNeuronT in inNeuronsTrain:
            self._indritesWt[inNeuronT]=self._indritesWtTrain[inNeuronT]

    def updateWeights(self):
        pass

    def setBiasWt(self,biasWt):
        self._biasWt = biasWt

    def printWtsTrain(self):
        wVals = ''
        #print('number of inbound Trains:',len(self._indritesWtTrain))
        for aNeuron in self._indritesWtTrain:
            wVals += ' ' + aNeuron.getLabel() + ':'
            wVal = "[{:.4f}]".format(self._indritesWtTrain[aNeuron])
            wVals += str(wVal)
        #print(wVals)

    def printWts(self):
        wVals = ''
        print('number of inbound connections:',len(self._indritesWt))
        for aNeuron in self._indritesWt:
            wVals += ' ' + aNeuron.getLabel() + ':'
            wVal = "[{:.4f}]".format(self._indritesWt[aNeuron])
            wVals += str(wVal)
        print(wVals)

    def printNeuron(self):
        print('neuron (Id,Label):'+str(self._id)+' - '+self._aid)
        print('num of indrites:' + str(len(self._indritesWt)))
        print('threshold:'+str(self._threashold))
        print('state:'+str(self._state))
        self.printWts()
        #self.printWtsTrain()

    def setState(self,onezero):
            self._state = onezero

    def setExpectedState(self,val):
            self._expectedState = val;

    def getState(self):
        return self._state

    def getExpectedState(self):
        return self._expectedState

    def getInputNeurons(self):
        return self._indritesWt

    def getInputNeuronsTrain(self):
        return self._indritesWtTrain

    def getLabel(self):
        return self._aid

    def GetJson(self):
        wVals = '  {'
        wVals += " \n   \"nrn\":\"" + self._aid + "\",\n"
        wVals += "   \"wts\":[  "
        for aNeuron in self._indritesWt:
            wVals += "{ \""+aNeuron.getLabel() + "\":"
            wVal = "{:.4f}".format(self._indritesWt[aNeuron])
            # wVal += ","
            wVals += str(wVal) + " },"
        wVals = wVals[:-1] # remove the last ',' char
        wVals += "  ]\n"
        wVals += "  }\n"
        return wVals