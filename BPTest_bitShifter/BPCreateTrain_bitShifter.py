from ..Algorithms.DL import Neuron as N,NeuralLayer as NL,NeuralNetwork as NN
import json as js

# C:\2_cPeaks\pyMusings>python -m VisuAlgorithms.CustomTests.BPCreateTrain_bitShifter

# create NN of 2 inputs, 3 hidden and 2 outputs

aNeuralNet = NN.NeuralNetwork()

n1 = N.Neuron(0,'IL_N1',None)
n2 = N.Neuron(1,'IL_N2',None)
n3 = N.Neuron(2,'IL_N3',None)
n4 = N.Neuron(3,'IL_N4',None)

il = NL.NeuralLayer()
il.setLayerBias(0.0)
lb = il.getLayerBias()
il.addNeuron(n1,lb)
il.addNeuron(n2,lb)
il.addNeuron(n3,lb)
il.addNeuron(n4,lb)


aNeuralNet.addLayerAt(1,il)
print('=====================================')


h1n1 = N.Neuron(4,'HL1_N1',{n1:0.15,n2:0.2,n3:0.2,n4:0.2})
h1n2 = N.Neuron(5,'HL1_N2',{n1:0.35,n2:0.64,n3:0.2,n4:0.2})
h1n3 = N.Neuron(6,'HL1_N3',{n1:0.5,n2:0.5,n3:0.2,n4:0.2})
h1n4 = N.Neuron(7,'HL1_N4',{n1:0.5,n2:0.5,n3:0.2,n4:0.2})
h1n5 = N.Neuron(8,'HL1_N5',{n1:0.5,n2:0.5,n3:0.2,n4:0.2})
h1n6 = N.Neuron(9,'HL1_N6',{n1:0.5,n2:0.5,n3:0.2,n4:0.2})

n1.setAxons([h1n1,h1n2,h1n3,h1n4,h1n5,h1n6])
n2.setAxons([h1n1,h1n2,h1n3,h1n4,h1n5,h1n6])
n3.setAxons([h1n1,h1n2,h1n3,h1n4,h1n5,h1n6])
n4.setAxons([h1n1,h1n2,h1n3,h1n4,h1n5,h1n6])



hl1 = NL.NeuralLayer()
hl1.setLayerBias(0.35)
lb = hl1.getLayerBias()
hl1.addNeuron(h1n1,lb)
hl1.addNeuron(h1n2,lb)
hl1.addNeuron(h1n3,lb)
hl1.addNeuron(h1n4,lb)
hl1.addNeuron(h1n5,lb)
hl1.addNeuron(h1n6,lb)


aNeuralNet.addLayerAt(2,hl1)

print('=====================================')


on1 = N.Neuron(10,'OL_N1',{h1n1:0.330,h1n2:0.24,h1n3:0.24,h1n4:0.24,h1n5:0.24,h1n6:0.24})
on2 = N.Neuron(11,'OL_N2',{h1n1:0.2320,h1n2:0.775,h1n3:0.24,h1n4:0.24,h1n5:0.24,h1n6:0.24})
on3 = N.Neuron(12,'OL_N3',{h1n1:0.330,h1n2:0.24,h1n3:0.24,h1n4:0.24,h1n5:0.24,h1n6:0.24})
on4 = N.Neuron(13,'OL_N4',{h1n1:0.2320,h1n2:0.775,h1n3:0.24,h1n4:0.24,h1n5:0.24,h1n6:0.24})

h1n1.setAxons([on1,on2,on3,on4])
h1n2.setAxons([on1,on2,on3,on4])
h1n3.setAxons([on1,on2,on3,on4])
h1n4.setAxons([on1,on2,on3,on4])
h1n5.setAxons([on1,on2,on3,on4])
h1n6.setAxons([on1,on2,on3,on4])

ol = NL.NeuralLayer()
ol.setLayerBias(0.60)
lb = ol.getLayerBias()
ol.addNeuron(on1,lb)
ol.addNeuron(on2,lb)
ol.addNeuron(on3,lb)
ol.addNeuron(on4,lb)

aNeuralNet.addLayerAt(3,ol)


aNeuralNet.PrintOutput()

# Train pattern batch
ioPatterns = [

    {'i': [1.0000, 0.0000, 0.0000, 0.0000], 'o':[0.0000, 0.0000, 0.0000, 1.0000]},
    {'i': [0.9000, 0.0001, 0.0006, 0.0006], 'o':[0.00043, 0.0006, 0.0006, 0.999]},
    {'i': [0.8900, 0.0001, 0.0006, 0.0006], 'o':[0.00023, 0.0006, 0.0006, 0.9678]},
    {'i': [0.7, 0.000132, 0.0006, 0.0006], 'o':[0.000132, 0.0006, 0.0006, 0.9965]},

    {'i': [0.0000, 1.0000, 0.0000, 0.0000], 'o':[1.0000, 0.0000, 0.0000, 0.0000]},
    {'i': [0.00043, 0.999, 0.0006, 0.0006], 'o':[0.999, 0.00043,  0.0006, 0.0006]},
    {'i': [0.00023, 0.9678, 0.0006, 0.0006], 'o':[0.9678, 0.00023,  0.0006, 0.0006]},
    {'i': [0.000132, 0.865, 0.0006, 0.0006], 'o': [0.865, 0.000132,  0.0006, 0.0006]},

    {'i': [0.0000, 0.0000, 1.0000, 0.0000], 'o': [0.0000, 1.0000, 0.0000, 0.0000]},
    {'i': [0.00043, 0.0006, 0.999, 0.0006], 'o': [0.00043, 0.999, 0.0006, 0.0006]},
    {'i': [0.00023, 0.0006, 0.9678, 0.0006], 'o': [0.00023, 0.9678, 0.0006, 0.0006]},
    {'i': [0.000132, 0.0006, 0.865, 0.0006], 'o': [0.000132, 0.865, 0.0006, 0.0006]},

    {'i': [0.0000, 0.0000, 0.0000, 1.0000], 'o': [0.0000, 0.0000, 1.0000, 0.0000]},
    {'i': [0.00043, 0.0006, 0.0006, 0.999], 'o': [0.00043, 0.0006, 0.999, 0.0006]},
    {'i': [0.00023, 0.0006, 0.0006, 0.9678], 'o': [0.00023, 0.0006, 0.9678, 0.0006]},
    {'i': [0.000132, 0.0006, 0.0006, 0.865], 'o': [0.000132, 0.0006, 0.865, 0.0006]},
]


aNeuralNet.TrainPatternBatch(ioPatterns)


#
#
# print("TRAIN 1 =================================================")
# input_float_array = [0.0, 1.0]
# output_float_array = [1.0, 0.0]
# aNeuralNet.TrainPattern(input_float_array, output_float_array)
# print("weights:------------------", aNeuralNet.GetJson())
# aNeuralNet.PrintOutput()
#
# print("TRAIN 2 =================================================")
# input_float_array = [0.0, 0.9]
# output_float_array = [1.1, 0.1]
# aNeuralNet.TrainPattern(input_float_array,output_float_array)
# print("weights:------------------",aNeuralNet.GetJson())
# # print("TRAIN 3 =================================================")
# # input_float_array = [1.0, 1.0]
# # output_float_array = [ 1.0, 0.0]
# # aNeuralNet.TrainPattern(input_float_array,output_float_array)
# # print("weights:------------------",aNeuralNet.GetJson())
# # aNeuralNet.PrintOutput()
# #
# # print("TRAIN 4 =================================================")
# # input_float_array = [0.0, 0.0]
# # output_float_array = [ 0.0, 1.0]
# # aNeuralNet.TrainPattern(input_float_array,output_float_array)
# # print("weights:------------------",aNeuralNet.GetJson())
# # aNeuralNet.PrintOutput()


print("TESTS =================================================")

aNeuralNet.RecognisePattern([0.9887, 0.00004, 0.00004, 0.00004])
aNeuralNet.RecognisePattern([0.0006, 0.00004, 0.97654, 0.00004])
aNeuralNet.RecognisePattern([ 0.00004,0.9887, 0.00004, 0.00004])
aNeuralNet.RecognisePattern([0.0006, 0.00004, 0.00004, 0.97654])


# aNeuralNet.RecognisePattern([0.0006, 0.00356])
# aNeuralNet.RecognisePattern([0.9887, 0.97654])


# let's try savine the NN model to a file

fileName = "bitShifter.txt" #asksaveasfilename()
jsonModel = ""

with open(fileName, "w") as outfile:
    jsonModel = aNeuralNet.GetJson()
    print(jsonModel)
    jsonMdl = js.loads(jsonModel)
    js.dump(jsonMdl, outfile)
    print("json:======:",jsonMdl)



