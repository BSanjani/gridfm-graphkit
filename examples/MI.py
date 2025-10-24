#import sys
#sys.path.append('~/graphs/GraphNNDabble/gridfm-graphkit/gridfm_graphkit')

from gridfm_graphkit.datasets.powergrid_dataset import GridDatasetDisk
from gridfm_graphkit.datasets.normalizers import BaseMVANormalizer
#from gridfm_graphkit.training.trainer import Trainer
from gridfm_graphkit.datasets.utils import split_dataset
from gridfm_graphkit.datasets.transforms import AddPFMask, AddRandomMask
from gridfm_graphkit.tasks.feature_reconstruction_task import FeatureReconstructionTask
from gridfm_graphkit.io.param_handler import NestedNamespace
#from gridfm_graphkit.utils.loss import PBELoss
from gridfm_graphkit.datasets.globals import PQ, PV, REF

# Standard open-source libraries
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from itertools import repeat
from .EvalUtils import DataStruct, Result, loadModel
import joblib
import pickle
import copy
import yaml
from datetime import datetime
from sklearn import svm
from sklearn.model_selection import train_test_split

temporary_storage='/dccstor/gridfm/sec_inf_temp/'
data_directory="/dccstor/gridfm/PowerGraph_TP/"


def plotAll(trainData,evalData,trainNames, evalNames, titles, suffix):
    #plot(trainData[3],evalData[3],trainNames,evalNames,titles[3],suffix)
    #plot(trainData[2],evalData[2],trainNames,evalNames,titles[2],suffix)
    #plot(trainData[0],evalData[0],trainNames,evalNames,titles[0],suffix)
    #plot(trainData[1],evalData[1],trainNames,evalNames,titles[1],suffix)
    for i in range(len(titles)):
        _ = plot(trainData[i],evalData[i],trainNames,evalNames,titles[i],suffix)

def plot(trainData,evalData,trainNames, evalNames, title, suffix):
    #print(trainData, evalData)
    plt.style.use('ggplot')
    fig = plt.figure()
    if len(trainData)>0:
        positionsTrain = range(len(trainNames))
        for pos in range(len(positionsTrain)):
            if len(trainData[pos])>1:
                #print([positionsTrain[pos]])
                plt.violinplot(trainData[pos],[positionsTrain[pos]])
    if len(evalData)>0:
        positionsEval = range(len(trainNames)+2,len(trainNames)+2+len(evalNames))
        #print(positionsEval, len(evalData))
        allLabels = trainNames+['']+evalNames
        #print(pos, allLabels)
        for pos in range(len(positionsEval)):
            if len(evalData[pos])>1:
                #print([positionsEval[pos]])
                plt.violinplot(evalData[pos],[positionsEval[pos]])
    pos = list(positionsTrain)+[len(trainNames)+1]+list(positionsEval)
    plt.xticks(pos, allLabels, rotation='vertical')
    plt.tight_layout()
    plt.savefig('plots/'+title+'_'+suffix+'.pdf')
    plt.close(fig)
    plt.clf()
    plt.cla()
    return 0

def addNanCorrect(tarlist, elem, removeNan):
    #print(elem,np.isnan(elem))
    if np.isnan(elem):
        if removeNan:
            pass
        else:
            tarlist.append(0.0)
    else:
        tarlist.append(elem)  
        #print('added', len(tarlist), tarlist[-1])      

def genData(dir, model,graph_wise=False,nodeWise=False,randomMask=False, removeNan=True,topologyOnly=False,debug=False):
    #diff = [[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2]]
    #diffVar =  [[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2]]
    #return diff, diffVar

    data_dir = data_directory+dir
    if debug:
        data_dir = 'data/case30_ieee'
    try:
        config_path = "config/"+data_dir.replace('data/','')+"base.yaml"
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    except:
        config_path = "config/case30_ieee_base.yaml"
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    config_args = NestedNamespace(**config_dict)
    node_normalizer, edge_normalizer = (
        BaseMVANormalizer(node_data=True,args=config_args),
        BaseMVANormalizer(node_data=False,args=config_args),
    )
    if randomMask:
        dataset = GridDatasetDisk(
            root=data_dir,
            norm_method="baseMVAnorm",
            node_normalizer=node_normalizer,
            edge_normalizer=edge_normalizer,
            pe_dim=20,  # Dimension of positional encoding
            transform=AddRandomMask(mask_dim=6,mask_ratio=0.5,args=config_args), #plus additional masks
        )
    else:
        dataset = GridDatasetDisk(
            root=data_dir,
            norm_method="baseMVAnorm",
            node_normalizer=node_normalizer,
            edge_normalizer=edge_normalizer,
            pe_dim=20,  # Dimension of positional encoding
            transform=AddPFMask(args=config_args),
        )

    # The scenarios are grouped in batches
    loader = DataLoader(dataset, batch_size=32)

    model.eval()
    if randomMask:
        diff = [[] for i in repeat(None, 18)]
        diffVar = [[] for i in repeat(None, 18)] 
    else:
        diff = [[] for i in repeat(None, 6)]
        diffVar = [[] for i in repeat(None, 6)] 

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)

            mask_PQ = batch.x[:, PQ] == 1 #PQ etc are just indexes
            mask_PV = batch.x[:, PV] == 1
            mask = batch.mask
            mask_REF = batch.x[:, REF] == 1
            # Apply random masking - not needed anymore due to new version of library
            #mask_value_expanded = model.mask_value.expand(batch.x.shape[0], -1)
            #batch.x[:, : batch.mask.shape[1]][batch.mask] = mask_value_expanded[batch.mask]
            #print(model.get_device())
            if topologyOnly:
                replacement = torch.zeros(batch.x.shape[0],6)
                batch.x[:,:6]=replacement
                #print(batch.x[:2,:]) #tested, works
                # Perform inference
                output = model(
                    batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch
                )
            else:
                # Perform inference
                output = model(
                    batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch
                )
            #filter out only nodes that it should predict
            ##iterate over all graphs and compute var/mean over prediction
            #print(output.shape) #960x6
            counter=0
            if torch.cuda.is_available():
                output = output.cpu()
                mask_PV = mask_PV.cpu()
                mask_PQ = mask_PV.cpu()
                batch = batch.cpu()
                mask_REF = mask_REF.cpu()
                mask = mask.cpu()
            if graph_wise:
                while counter in batch.batch:
                    error = np.sqrt((output[batch.batch==counter]-batch.y[batch.batch==counter])**2)
                    if randomMask:    
                        combined_mask = torch.logical_and(mask_PQ[batch.batch==counter],mask[batch.batch==counter])
                        var, mean = torch.var_mean(error[combined_mask],dim=0)
                        for i, j in zip([6,7,8,9], [0,1,2,3]):
                            addNanCorrect(diff[i], mean[j].item(), removeNan)
                            addNanCorrect(diffVar[i], var[j].item(), removeNan)
                    else:
                        var, mean = torch.var_mean(error[mask_PQ[batch.batch==counter]],dim=0)
                    addNanCorrect(diff[0], mean[4].item(), removeNan)
                    addNanCorrect(diffVar[0], var[4].item(), removeNan)
                    addNanCorrect(diff[1], mean[5].item(), removeNan)
                    addNanCorrect(diffVar[1], var[5].item(), removeNan)
                    if randomMask:    
                        combined_mask = torch.logical_and(mask_PV[batch.batch==counter],mask[batch.batch==counter])
                        var, mean = torch.var_mean(error[combined_mask],dim=0)
                        for i, j in zip([10,11,12,13], [0,1,2,4]):
                            addNanCorrect(diff[i], mean[j].item(), removeNan)
                            addNanCorrect(diffVar[i], var[j].item(), removeNan)
                    else:
                        var, mean = torch.var_mean(error[mask_PV[batch.batch==counter]],dim=0)
                    addNanCorrect(diff[2], mean[3].item(), removeNan)
                    addNanCorrect(diffVar[2], var[3].item(), removeNan)
                    addNanCorrect(diff[3], mean[5].item(), removeNan)
                    addNanCorrect(diffVar[3], var[5].item(), removeNan)
                    if randomMask:    
                        combined_mask = torch.logical_and(mask_REF[batch.batch==counter],mask[batch.batch==counter])
                        var, mean = torch.var_mean(error[combined_mask],dim=0)
                        for i, j in zip([14,15,16,17], [0,1,4,5]):
                            addNanCorrect(diff[i], mean[j].item(), removeNan)
                            addNanCorrect(diffVar[i], var[j].item(), removeNan)
                    else:
                        var, mean = torch.var_mean(error[mask_REF[batch.batch==counter]],dim=0)
                    addNanCorrect(diff[4], mean[2].item(), removeNan)
                    addNanCorrect(diffVar[4], var[2].item(), removeNan)
                    addNanCorrect(diff[5], mean[3].item(), removeNan)
                    addNanCorrect(diffVar[5], var[3].item(), removeNan)
                    counter = counter+1 
            elif nodeWise: #compute per node
                error = np.sqrt((output-batch.y)**2)
                ###PQ
                if randomMask:
                    combined_mask = torch.logical_and(mask_PQ,mask)
                    tempRes=torch.nan_to_num(copy.copy(error[combined_mask,:]))
                    for i, j in zip([6,7,8,9], [0,1,2,3]):
                        diff[i]=diff[i]+tempRes[:,j].tolist()
                else:
                    tempRes = torch.nan_to_num(copy.copy(error[mask_PQ,:]))
                    diff[0]= diff[0]+tempRes[:,4].tolist()
                    diff[1] = diff[1]+tempRes[:,5].tolist()
                #### PV
                if randomMask:
                    combined_mask = torch.logical_and(mask_PV,mask)
                    tempRes=torch.nan_to_num(copy.copy(error[combined_mask,:]))
                    for i, j in zip([10,11,12,13], [0,1,2,4]):
                        diff[i]=diff[i]+tempRes[:,j].tolist()
                else: 
                    tempRes = torch.nan_to_num(copy.copy(error[mask_PV,:]))
                    diff[2]= diff[2]+tempRes[:,3].tolist()
                    diff[3] = diff[3]+tempRes[:,5].tolist()
                #REF
                if randomMask:
                    combined_mask = torch.logical_and(mask_REF,mask)
                    tempRes=torch.nan_to_num(copy.copy(error[combined_mask,:]))
                    for i, j in zip([14,15,16,17], [0,1,4,5]):
                        diff[i]=diff[i]+tempRes[:,j].tolist()
                else:
                    var, mean = torch.var_mean(error[mask_REF,:],dim=0) 
                    diff[4]= diff[4]+tempRes[:,2].tolist()
                    diff[5] = diff[5]+tempRes[:,3].tolist() 
            else: #compute per batch
                error = np.sqrt((output-batch.y)**2)
                #PQ
                if randomMask:
                    combined_mask = torch.logical_and(mask_PQ,mask)
                    var, mean = torch.var_mean(error[combined_mask,:],dim=0)
                    for i, j in zip([6,7,8,9], [0,1,2,3]):
                        addNanCorrect(diff[i], mean[j].item(), removeNan)
                        addNanCorrect(diffVar[i], var[j].item(), removeNan)
                else:
                    var, mean = torch.var_mean(error[mask_PQ,:],dim=0)
                addNanCorrect(diff[0], mean[4].item(), removeNan)
                addNanCorrect(diffVar[0], var[4].item(), removeNan)
                addNanCorrect(diff[1], mean[5].item(), removeNan)
                addNanCorrect(diffVar[1], var[5].item(), removeNan)
                #PV
                if randomMask:
                    combined_mask = torch.logical_and(mask_PV,mask)
                    var, mean = torch.var_mean(error[combined_mask,:],dim=0)
                    for i, j in zip([10,11,12,13], [0,1,2,4]):
                        addNanCorrect(diff[i], mean[j].item(), removeNan)
                        addNanCorrect(diffVar[i], var[j].item(), removeNan)
                else:
                    var, mean = torch.var_mean(error[mask_PV,:],dim=0)   
                addNanCorrect(diff[2], mean[3].item(), removeNan)
                addNanCorrect(diffVar[2], var[3].item(), removeNan)
                addNanCorrect(diff[3], mean[5].item(), removeNan)
                addNanCorrect(diffVar[3], var[5].item(), removeNan)  
                #REF
                if randomMask:
                    combined_mask = torch.logical_and(mask_REF,mask)
                    var, mean = torch.var_mean(error[combined_mask,:],dim=0)
                    for i, j in zip([14,15,16,17], [0,1,4,5]):
                        addNanCorrect(diff[i], mean[j].item(), removeNan)
                        addNanCorrect(diffVar[i], var[j].item(), removeNan)
                else:
                    var, mean = torch.var_mean(error[mask_REF,:],dim=0)  
                addNanCorrect(diff[4], mean[2].item(), removeNan)
                addNanCorrect(diffVar[4], var[2].item(), removeNan)
                addNanCorrect(diff[5], mean[3].item(), removeNan)
                addNanCorrect(diffVar[5], var[3].item(), removeNan)  
    #for i in range(len(diff)):
    #    print(len(diff[i]),len(diffVar[i]))
    return diff, diffVar  

def evalMIallFeatures(trainData, testData,titles, suffix, blackbox=True,threads=4):
   MembershipBlackBox(trainData,testData, 'All', suffix, dim=len(trainData[0])) 

def evalMI(trainData, testData,titles, suffix, blackbox=True,threads=4):
    #iters = int(((len(titles)+1.0)/threads)+0.9)
    #tempind = np.array_split(np.arange(len(titles)),iters)
    #print(iters, tempind, len(trainData),len(trainData[0]))
    #for index in range(iters):
    #    with Pool(threads) as p:
    #        params = list(zip(trainData[tempind[index]], testData[tempind[index]], np.array(titles)[tempind[index]], [suffix]*np.shape(tempind[index])[0]))
    #        p.starmap(MembershipBlackBox, params)
    #        #p.starmap(debugMIDummy, params)
    for i in range(len(titles)):
        MembershipBlackBox(trainData[i],testData[i],titles[i],suffix)

def fitSVM(data, labels):
    if np.shape(data)[0]>700000:
        #for runtime reasons, we only train on max 90,000 samples
        # per = 70000.0/float(np.shape(data)[0]) #previously
        per = 1.0-(50000.0/float(np.shape(data)[0]))
        X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=per, random_state=42)
        #we also don't test on all remaining samples
        per = 15000.0/float(np.shape(X_temp)[0])
        _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=per, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    bestPerf = 0.0
    bestclf = None
    bestkernel = ''
    bestc = 0
    bestgamma = 0
    for c in  [0.01, 0.1, 1, 10, 100]: #removed 0.001
        clf = svm.SVC(C=c, kernel='linear')
        clf.fit(X_train,y_train)
        res = clf.score(X_test,y_test)
        if res > bestPerf:
            bestPerf = res
            bestclf = clf
            bestc =c
            bestgamma = -1.0
            bestkernel = 'linear'
    for gamma in [0.001, 0.01, 0.1, 1,10,100]: #removed 0.0001 due to poor performance
        for c in  [0.01, 0.1, 1, 10, 100]: #removed 0.001
            clf = svm.SVC(C=c, kernel='rbf', gamma=gamma)
            clf.fit(X_train,y_train)
            res = clf.score(X_test,y_test)
            if res > bestPerf:
                bestPerf = res
                bestclf = clf
                bestc =c
                bestgamma = gamma
                bestkernel = 'rbf'
    return bestclf, bestPerf, 'c '+str(bestc), bestgamma, bestkernel

def recombine(arr, dim=1,index=-1):
    if dim==1:
        combined = []
        for elem in arr:
            combined = combined+elem
    else:
        combined = [[] for i in repeat(None, dim)]
        if index>=0:
            for i in range(dim):
                combined[i]=combined[i]+arr[i][index]
        else:
            for i in range(dim):
                for elem in arr[i]:
                    combined[i]=combined[i]+elem
    return combined


def MembershipBlackBox(trainData,testData, title, suffix, dim=1):
    res = []
    # on full data - worst case / theoretically achievable
    fulltrain = recombine(trainData,dim)
    fullTest = recombine(testData,dim)
    if len(fulltrain)==0 or len(fullTest)==0:
        return None
    #compute over all datasets
    if dim==1:
        labs = getLabels(len(fulltrain),len(fullTest))
        data = np.array(fulltrain+fullTest).reshape(-1,dim)
    else:
        data = np.hstack((np.array(fulltrain),np.array(fullTest)))
        data = np.transpose(data)
        labs = getLabels(np.shape(np.array(fulltrain))[1],np.shape(np.array(fullTest))[1])
    clf, fullScore, c, gamma, kernel = fitSVM(data, labs)
    joblib.dump(clf, temporary_storage+'SVM/'+title+suffix+'_SVM.pickle')
    baselineAcc = float(np.mean(labs))
    fas = getFalseAlarmRate(clf,data,labs)
    res = DataStruct(fullScore, baselineAcc, fas, c, gamma, kernel)
    # individual datasets
    for i in range(len(trainData)):
        for j in range(len(testData)):
            if dim ==1:
                labs = getLabels(len(trainData[i]),len(testData[j]))
                data = np.array(trainData[i]+testData[j]).reshape(-1,dim)  
            else:
                trainD = recombine(trainData,dim,i)
                testD = recombine(testData,dim,j)
                #print(np.shape(trainD),np.shape(testD))
                data = np.hstack((np.array(trainD),np.array(testD)))
                data = np.transpose(data)
                labs = getLabels(np.shape(trainD)[1],np.shape(testD)[1])
            clf, fullScore, c, gamma, kernel = fitSVM(data, labs)
            indRes = Result(fullScore, c, gamma, kernel, i, j)
            for k in range(len(trainData)):
                if k==i:
                    pass       
                else:
                    if dim==1:
                        parScore = clf.score(np.array(trainData[k]).reshape(-1,dim),getLabels(len(trainData[k]),0))
                    else:
                        trainTemp = recombine(trainData,dim,k)
                        parScore = clf.score(np.array(trainTemp).reshape(-1,dim),getLabels(np.shape(trainTemp)[1],0))
                    indRes.addTrainResult(parScore)
            for k in range(len(testData)):
                if k==j:
                    pass
                else:
                    if dim==1:
                        parScore = clf.score(np.array(testData[k]).reshape(-1,dim),getLabels(0,len(testData[k])))
                        fas = getFalseAlarmRate(clf,np.array(testData[k]).reshape(-1,dim),getLabels(0,len(testData[k])))
                    else:
                        testTemp = recombine(testData,dim,k)
                        parScore = clf.score(np.array(testTemp).reshape(-1,dim),getLabels(np.shape(testTemp)[1],0))
                        fas = getFalseAlarmRate(clf,np.array(testTemp).reshape(-1,dim),getLabels(np.shape(testTemp)[1],0))
                    indRes.addTestResult(parScore)
                    indRes.addFASResult(fas)
            res.addResult(indRes)
    # write output setting
    print(title, suffix)
    print(res.settingsStats())
    res.summarize(name=title+suffix, printRes=True)
    # save plot to analyse performance
    res.plot(title, suffix+'_MI')
    with open(temporary_storage+'interRes/'+title+suffix+'MI.pickle', 'wb') as f:
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    #res.serialize(temporary_storage+'interRes/',title,suffix+'MI.pickle')

# This network was chosen for visualization purposes (networks up to 300 buses were tested)
# The number of load scenarios is 1024

def debugMI(graphWise, model12, randomMask, topologyOnly, featureNames):
    if model12:
        model = loadModel("models/GridFM_v0_1.pth")
    else:
        model = loadModel("models/GridFM_v0_2.pth")
    #non-training data
    if randomMask:
        evalMean = [[] for i in repeat(None, 18)]
        evalVar = [[] for i in repeat(None, 18)]  
    else:
        evalMean = [[] for i in repeat(None, 6)]
        evalVar = [[] for i in repeat(None, 6)] 
    mean, var = genData('case14_ieee', model,graph_wise=graphWise,randomMask=randomMask)
    #print('generated data',len(mean),len(mean[0]))
    for i in range(len(mean)):
        evalMean[i].append(mean[i])
        evalMean[i].append(mean[i])
    if randomMask:
        trainMean = [[] for i in repeat(None, 18)]
        trainVar = [[] for i in repeat(None, 18)] 
    else:
        trainMean = [[] for i in repeat(None, 6)]
        trainVar = [[] for i in repeat(None, 6)] 
    mean, var = genData('case30_ieee', model,graph_wise=graphWise,randomMask=randomMask)
    #print('generated train data',len(mean),len(mean[0]))
    for i in range(len(mean)):
        trainMean[i].append(mean[i])
        trainMean[i].append(mean[i])
    evalMI(evalMean,trainMean,['14','14','30','30'], 'DEBUG2')

def genCasesSVMCrossEval(case='mean', featureList=[]):
    params  = [[True, True, True],[True, True, False],[True, False, True],[True, False, False],
               [False, True, True],[False, True, False],[False, False, True],[False, False, False]]
    removeNan = False
    randomMask= False
    cases = []
    for elem in params:
        suffix = ''#'All_'
        if elem[0]:
            suffix = suffix+'GW_'
        if elem[1]:
            suffix = suffix+'M12_'
        if randomMask:
            suffix = suffix+'R_'
        if elem[2]:
            suffix = suffix+'topolOnly_'
        if not removeNan:
            suffix = suffix+'NaNZ_'
        cases.append(suffix)
    results = np.ones((2,len(cases),len(cases)))*-1.0
    for i in range(len(cases)):
        trainData = []
        evalData = []
        maxIter = 6
        dataDirTest = dataDirTrain = ''
        try:
            for feature in featureList:
                if 'ea' in case:
                    dataDirTrain = temporary_storage+feature+cases[i]+'meanTrainTemp.pickle'
                else:
                    dataDirTrain = temporary_storage+feature+cases[i]+'varTrainTemp.pickle'
                with open(dataDirTrain, 'rb') as f:
                    trainData.append(pickle.load(f))
                if 'ea' in case:
                    dataDirTest = temporary_storage+feature+cases[i]+'meanEvalTemp.pickle'
                else:
                    dataDirTest = temporary_storage+feature+cases[i]+'varTrainTemp.pickle'    
                with open(dataDirTest, 'rb') as f:
                    evalData.append(pickle.load(f))
            #sanity check Length - not neccessary, loading for feature would fail
            fulltrain = recombine(trainData,6)
            fullTest = recombine(evalData,6)
            data = np.hstack((np.array(fulltrain),np.array(fullTest)))
            data = np.transpose(data)
            labs = getLabels(np.shape(np.array(fulltrain))[1],np.shape(np.array(fullTest))[1])
            if np.shape(data)[0]>700000:
                per = 50000.0/float(np.shape(data)[0])
                _, data, _, labs = train_test_split(data, labs, test_size=per, random_state=42)
            print(np.shape(data),np.shape(labs))
            for j in range(i, len(cases)):
                try:
                    if 'ea' in case:
                        svmDir = temporary_storage+'SVM/All'+cases[j]+'mean_SVM.pickle'
                    else:
                        svmDir = temporary_storage+'SVM/All'+cases[j]+'var_SVM.pickle'
                    clf = joblib.load(svmDir)
                    res =  clf.score(data,labs)
                    fas = getFalseAlarmRate(clf,data,labs)
                    results[0,i,j]=results[0,j,i]=res
                    results[1,i,j]=results[1,j,i]=fas
                except FileNotFoundError:
                    print(svmDir+' not found!')                    
        except FileNotFoundError:
            print(dataDirTest+' or '+dataDirTrain+' not found!')
    ##add data loading??
    print(cases)
    print(results[0])
    print(results[1])
      

def mainExperiments(graphWise,nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot, MI, runIndividualFeatures=False,step='load',data_dir_train=[],data_dir_eval=[]):
    #print('started',graphWise,nodeWise, model12,topologyOnly)
    data_dir = temporary_storage
    evalNames = ['39 epri','60c','1354','197 snem','300','73 rts','14','5 pfm']
    trainNames = ['240 pserc','24 ieee','57 ieee','89 pegase','118 ieee','30']
    suffix = ''
    if graphWise:
        suffix = suffix+'GW_'
    if nodeWise:
        suffix = suffix+'NW_'
    if model12:
        suffix = suffix+'M12_'
    if randomMask:
        suffix = suffix+'R_'
    if topologyOnly:
        suffix = suffix+'topolOnly_'
    if not removeNan:
        suffix = suffix+'NaNZ_'
    if 'gen' in step:        ### check whether data is already there
        try:
            with open(data_dir+featureNames[0]+suffix+'meanTrainTemp.pickle', 'rb') as f:
                _ = pickle.load(f)
            #step = 'schr'
            print(suffix,' Data already generated!')
        except FileNotFoundError:
            pass
    if 'gen' in step:
        if model12:
            model = loadModel("models/GridFM_v0_1.pth")
        else:
            model = loadModel("models/GridFM_v0_2.pth")
        if torch.cuda.is_available():
            #move model to gpu
            model.model.to('cuda')
        if randomMask:
            evalMean = [[] for i in repeat(None, 18)]
            evalVar = [[] for i in repeat(None, 18)] 
        else:
            evalMean = [[] for i in repeat(None, 6)]
            evalVar = [[] for i in repeat(None, 6)]
        if len(data_dir_eval)<1:
            data_dirs = ['case39_epri','case60_c','case1354_pegase','case197_snem','case300_ieee','case73_ieee_rts','case14_ieee','case5_pjm']
        else:
            data_dirs = data_dir_eval
        for datadir in data_dirs:
            mean, var = genData(datadir, model,graph_wise=graphWise,nodeWise=nodeWise,randomMask=randomMask,topologyOnly=topologyOnly, removeNan=removeNan)
            for i in range(len(mean)):
                evalMean[i].append(mean[i])
                if not nodeWise:
                    evalVar[i].append(var[i])
        if len(data_dir_train)<1:
            data_dirs = ['case240_pserc','case24_ieee_rts', 'case57_ieee','case89_pegase','case118_ieee','case30_ieee']
        else:
            data_dirs = data_dir_train
        if randomMask:
            trainMean = [[] for i in repeat(None, 18)]
            trainVar = [[] for i in repeat(None, 18)] 
        else:
            trainMean = [[] for i in repeat(None, 6)]
            trainVar = [[] for i in repeat(None, 6)]
        for datadir in data_dirs:
            mean, var = genData(datadir, model,graph_wise=graphWise,nodeWise=nodeWise,randomMask=randomMask,topologyOnly=topologyOnly, removeNan=removeNan)
            for i in range(len(mean)):
                trainMean[i].append(mean[i])
                if not nodeWise:
                    trainVar[i].append(var[i])
        #serialize
        if 'serial' in step:
            #train mean, train var, test mean, test var
            maxIter = 6
            if randomMask: 
                maxIter=18
            data_dir = temporary_storage
            for i in range(maxIter):
                print(data_dir)
                print('saving',data_dir+featureNames[i]+suffix+'meanTrainTemp.pickle')
                with open(data_dir+featureNames[i]+suffix+'meanTrainTemp.pickle', 'wb') as f:
                    # Pickle the arrays using the highest protocol available.
                    pickle.dump(trainMean[i], f, pickle.HIGHEST_PROTOCOL)
                with open(data_dir+featureNames[i]+suffix+'meanEvalTemp.pickle', 'wb') as f:
                    # Pickle the arrays using the highest protocol available.
                    pickle.dump(evalMean[i], f, pickle.HIGHEST_PROTOCOL)
                if not nodeWise:
                    with open(data_dir+featureNames[i]+suffix+'varTrainTemp.pickle', 'wb') as f:
                        # Pickle the arrays using the highest protocol available.
                        pickle.dump(trainVar[i], f, pickle.HIGHEST_PROTOCOL)
                    with open(data_dir+featureNames[i]+suffix+'varEvalTemp.pickle', 'wb') as f:
                        # Pickle the arrays using the highest protocol available.
                        pickle.dump(evalVar[i], f, pickle.HIGHEST_PROTOCOL)
    #desearialize
    if 'load' in step:
        trainMean = []
        trainVar = []
        evalMean = []
        evalVar = []
        maxIter = 6
        if randomMask: 
            maxIter=18
        try:
            data_dir = temporary_storage
            for i in range(maxIter):
                with open(data_dir+featureNames[i]+suffix+'meanTrainTemp.pickle', 'rb') as f:
                    trainMean.append(pickle.load(f))
                with open(data_dir+featureNames[i]+suffix+'meanEvalTemp.pickle', 'rb') as f:
                    evalMean.append(pickle.load(f))
                if not nodeWise:
                    with open(data_dir+featureNames[i]+suffix+'varEvalTemp.pickle', 'rb') as f:
                        evalVar.append(pickle.load(f))
                    with open(data_dir+featureNames[i]+suffix+'varTrainTemp.pickle', 'rb') as f:
                        trainVar.append(pickle.load(f))
            #sanity check Length - not neccessary, loading for feature would fail
        except FileNotFoundError:
            print(featureNames[i]+suffix, 'NOT FOUND!!')
            plot = False
            MI = False
    if plot:
        plotAll(trainMean,evalMean,trainNames, evalNames, featureNames, suffix+'mean')
        if not nodeWise:
            plotAll(trainVar,evalVar,trainNames, evalNames, featureNames, suffix+'var')
    if MI:
        #check membership
        if not removeNan and not runIndividualFeatures:
            evalMIallFeatures(trainMean,evalMean, featureNames, suffix+'mean')
            if not nodeWise:
                evalMIallFeatures(trainVar,evalVar, featureNames, suffix+'var')
        if runIndividualFeatures:
            evalMI(trainMean,evalMean, featureNames, suffix+'mean')
            if not nodeWise:
                evalMI(trainVar,evalVar, featureNames, suffix+'var') 

