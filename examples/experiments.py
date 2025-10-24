
from .MI import mainExperiments, genData, plotAll, evalMI
from multiprocessing import Pool
import torch
import random
from .EvalUtils import loadModel
from .EvalUtils import getLabels, getFalseAlarmRate
from itertools import repeat

torch.manual_seed(0)
random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debugPlots(graphWise, model12, randomMask, topologyOnly, featureNames):
    if model12:
        model = loadModel("models/GridFM_v0_1.pth")
    else:
        model = loadModel("models/GridFM_v0_2.pth")
    'case14_ieee'
    mean, var = genData('',model,graph_wise=graphWise, debug=True,randomMask=False,topologyOnly=topologyOnly)
    #restructure data
    for i in range(len(mean)):
        print(len(mean[i]),len(var[i]))
    if randomMask:
        evalDMean = [[] for i in repeat(None, 18)]
        evalDVar = [[] for i in repeat(None, 18)] 
    else:
        evalDMean = [[] for i in repeat(None, 6)]
        evalDVar = [[] for i in repeat(None, 6)] 
    for j in range(len(mean)):
        for i in range(3):
            evalDMean[j].append(mean[j])
            if i<2:
                evalDVar[j].append(var[j])
    for i in range(len(evalDMean)):
        print(len(evalDMean[i]),len(evalDVar[i]))
    plotAll(evalDMean,evalDVar,['t1','t2','t3'], ['v1','v2'], featureNames, 'testNode')
    #evalMI(evalDMean,evalDVar,featureNames, 'DEBUG')

def debugMIDummy(parama, paramb, paramc, paramd):
    print(parama, paramb,paramc,paramd)

def runDist(graphWise, model12,topologyOnly,nodeWise=False,gen=False):
    removeNan = False
    randomMask= False
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    if randomMask:
        featureNames= featureNames+['PQ-ActivePowerDemand','PQ-ReactivePowerDemand','PQ-ActivePowerGenerated','PQ-ReactivePowerGenerated', 'PV-ActivePowerDemand','PV-ReactivePowerDemand','PV-ReactivePowerGenerated', 'PV-VoltageMagnitude','REF-ActivePowerDemand','REF-ReactivePowerDemand', 'REF-VoltageMagnitude','REF-VoltageAngle']
    if gen:
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,step='genserial')
    else:
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,step='load')


def runExperiments(threads = 4,case=1):
    if case==1:
        params  = iter([[True, True, True],[True, True, False],[True, False, True],[True, False, False]])
    else:
        params  = iter([[False, True, True],[False, True, False],[False, False, True],[False, False, False]])
    with Pool(threads) as p:
        p.starmap(runDist, params)

def runExperimentsData():
    params  = [[True, True, True],[True, True, False],[True, False, True],[True, False, False],
               [False, True, True],[False, True, False],[False, False, True],[False, False, False]]
    for elem in params:
        runDist(elem[0],elem[1],elem[2])

def runExperimentsNodeWise(gen=False):
    graphWise = False
    nodeWise = True
    model12 = False
    randomMask= False
    topologyOnly=False
    removeNan = False
    #data_dirs_eval = [['case39_epri','case60_c'],['case1354_pegase','case197_snem'],['case300_ieee','case73_ieee_rts'],['case14_ieee','case5_pjm']]
    #data_dirs_train = [['case240_pserc'],['case24_ieee_rts', 'case57_ieee'],['case89_pegase','case118_ieee'],['case30_ieee']]      
    if gen:
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='genserial')
        topologyOnly=True
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='genserial')
    else:
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='load')
        topologyOnly=True
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='load')
    
def runExperimentsSingleFeature():
    graphWise = False
    nodeWise = False
    model12 = False
    randomMask= False
    topologyOnly=False
    removeNan = False
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='load')
    topologyOnly=True
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='load')
    
def runGraphWise():
    runDist(True, True, True)

def runMissing():
    runDist(False, True, True)
    runDist(True, True, True)

def runSmallTopology():
    graphWise = False
    nodeWise = False
    model12 = True
    randomMask= False
    topologyOnly=True
    removeNan = False
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial')
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load')
    

if __name__ == "__main__":
    graphWise = False
    nodeWise = True
    model12 = False
    randomMask= False
    topologyOnly=False
    removeNan = False

    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    if randomMask:
        featureNames = featureNames+['PQ-ActivePowerDemand','PQ-ReactivePowerDemand','PQ-ActivePowerGenerated','PQ-ReactivePowerGenerated', 'PV-ActivePowerDemand','PV-ReactivePowerDemand','PV-ReactivePowerGenerated', 'PV-VoltageMagnitude','REF-ActivePowerDemand','REF-ReactivePowerDemand', 'REF-VoltageMagnitude','REF-VoltageAngle']
    #featureNames = ['Active Power Demand','Reactive Power Demand','Active Power Generated','Reactive Power Generated', 'Voltage Magnitude','Voltage Angle','PQ','PV','REF']

    #debugPlots(graphWise, model12, randomMask, topologyOnly, featureNames)
    #debugMI(graphWise, model12, randomMask, topologyOnly, featureNames)
    #mainExperiments(graphWise, nodeWise,model12, randomMask, topologyOnly, featureNames, removeNan, plot=True, MI=False,step='genserial')
    #mainExperiments(graphWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,step='load')
    #evalMI(['a','b','c','d','e','f','g'], ['1','2','3','4','5','6','7'],featureNames, 'Testuffix')
    #runMissing()
    #runExperiments(case=1)
    #genCasesSVMCrossEval(featureList=featureNames)
    #runExperimentsSingleFeature()
    #runExperimentsNodeWise()
    #runDist(False,False, False)
    #runMissing()
    #runExperimentsData()
    #runDist(False,True,True)
    #runDist(True,True,True)
    #runSmallTopology()
    #runGraphWise()
    #GW: (2417043, 6) -> Too large
    #BW:   (75541, 6) -> Fine (with test 0.2 = ~14000)









