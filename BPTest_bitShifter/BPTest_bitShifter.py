from ..Algorithms.DL import Neuron as N,NeuralLayer as NL,NeuralNetwork as NN
from ..DataStructures.cpEnums import *
import json as js

# C:\2_cPeaks\pyMusings>python -m VisuAlgorithms.CustomTests.BPTest_bitShifter


fileName = "bitShifter.txt" #asksaveasfilename()
jsonModel = ""

aNeuralNet = NN.NeuralNetwork.BuildModelFromJsonFile(fileName)
aNeuralNet.printNetwork()


aNeuralNet.RecognisePattern([0.9887, 0.00004, 0.00004, 0.00004])
aNeuralNet.RecognisePattern([0.0006,0.0006,0.0006, 0.97654])

aNeuralNet.RecognisePattern([0.9887, 0.00004, 0.00004, 0.00004])
aNeuralNet.RecognisePattern([0.0006, 0.00004, 0.97654, 0.00004])
aNeuralNet.RecognisePattern([ 0.00004,0.9887, 0.00004, 0.00004])
aNeuralNet.RecognisePattern([0.0006, 0.00004, 0.00004, 0.97654])



