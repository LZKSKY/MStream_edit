import random
import os
import time
import json
import copy

class Model:
    KIncrement = 100
    smallDouble = 1e-150
    largeDouble = 1e150
    Max_Batch = 5 # Max number of batches we will consider

    def __init__(self, K, KIncrement, V, iterNum,alpha, beta, dataset, ParametersStr, sampleNo, wordsInTopicNum, timefil):
        self.dataset = copy.deepcopy(dataset)
        self.ParametersStr = copy.deepcopy(ParametersStr)
        self.alpha = copy.deepcopy(alpha)
        self.beta = copy.deepcopy(beta)
        self.K = copy.deepcopy(K)
        self.Kin = copy.deepcopy(K)
        self.V = copy.deepcopy(V)
        self.iterNum = copy.deepcopy(iterNum)
        self.beta0 = copy.deepcopy(float(V) * float(beta))
        self.KIncrement = copy.deepcopy(KIncrement)
        self.Kmax = copy.deepcopy(max(K, KIncrement))
        self.sampleNo = copy.deepcopy(sampleNo)
        self.wordsInTopicNum = copy.deepcopy(wordsInTopicNum)
        self.phi_zv = []

        self.batchNum2tweetID = {}  # batch num to tweet id
        self.batchNum = 1  # current batch number

        '''
        with open(timefil) as timef:
            for line in timef:
                buff = line.strip().split(' ')
                if buff == ['']:
                    break
                self.batchNum2tweetID[self.batchNum] = int(buff[1])
                self.batchNum += 1
        self.batchNum = 1
        print("There are", self.batchNum2tweetID.__len__(), "time points.\n\t", self.batchNum2tweetID)
        '''

    def run(self, documentSet, outputPath, wordList):
        self.D_All = documentSet.D  # The whole number of documents
        self.z = {}  # Cluster assignments of each document                 (documentID -> clusterID)
        self.m_z = {}  # The number of documents in cluster z               (clusterID -> number of documents)
        self.n_z = {}  # The number of words in cluster z                   (clusterID -> number of words)
        self.n_zv = {}  # The number of occurrences of word v in cluster z  (n_zv[clusterID][wordID] = number)
        self.currentDoc = 0  # Store start point of next batch
        self.startDoc = 0  # Store start point of this batch
        self.D = 0  # The number of documents currently
        self.K_current = copy.deepcopy(self.K) # the number of cluster containing documents currently
        self.BatchSet = {} # Store information of each batch
        self.DocInBatch = 2000
        while self.currentDoc < self.D_All:
            print("Batch", self.batchNum)
            '''
            if self.batchNum not in self.batchNum2tweetID:
                break
            '''
            if self.batchNum <= self.Max_Batch:
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)
                self.intialize(documentSet)
                self.gibbsSampling(documentSet)
            else:
                # remove influence of batch earlier than Max_Batch
                self.D -= self.BatchSet[self.batchNum - self.Max_Batch]['D']
                for cluster in self.m_z:
                    if cluster in self.BatchSet[self.batchNum - self.Max_Batch]['m_z']:
                        self.m_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['m_z'][cluster]
                        self.checkEmpty(cluster)
                        self.n_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['n_z'][cluster]
                        for word in self.n_zv[cluster]:
                            if word in self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster]:
                                self.n_zv[cluster][word] -= self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster][word]
                self.BatchSet.pop(self.batchNum - self.Max_Batch)
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)
                self.intialize(documentSet)
                self.gibbsSampling(documentSet)
            # get influence of only the current batch (remove other influence)
            self.BatchSet[self.batchNum-1]['D'] = self.D - self.BatchSet[self.batchNum-1]['D']
            for cluster in self.m_z:
                if cluster not in self.BatchSet[self.batchNum - 1]['m_z']:
                    self.BatchSet[self.batchNum - 1]['m_z'][cluster] = 0
                if cluster not in self.BatchSet[self.batchNum - 1]['n_z']:
                    self.BatchSet[self.batchNum - 1]['n_z'][cluster] = 0
                self.BatchSet[self.batchNum - 1]['m_z'][cluster] = self.m_z[cluster] - self.BatchSet[self.batchNum - 1]['m_z'][cluster]
                self.BatchSet[self.batchNum - 1]['n_z'][cluster] = self.n_z[cluster] - self.BatchSet[self.batchNum - 1]['n_z'][cluster]
                if cluster not in self.BatchSet[self.batchNum - 1]['n_zv']:
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster] = {}
                for word in self.n_zv[cluster]:
                    if word not in self.BatchSet[self.batchNum - 1]['n_zv'][cluster]:
                        self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = 0
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = self.n_zv[cluster][word] - self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word]
            print("\tGibbs sampling successful! Start to saving results.")
            self.output(documentSet, outputPath, wordList, self.batchNum - 1)
            print("\tSaving successful!")
            print('\tSaving probablity')
            self.save_pro_iter(self.batchNum, self.iterNum, self.pro_cal)
            print('\tbatch ' + str(self.batchNum) + ' iterNum ' + str(self.iterNum) + 'saved')

    def intialize(self, documentSet):
        '''
        for d in range(self.currentDoc, self.D_All):
            documentID = documentSet.documents[d].documentID
            if documentID <= self.batchNum2tweetID[self.batchNum]:
                self.D += 1
            else:
                break
        '''
        self.startDoc = self.currentDoc
        self.currentDoc = min(self.currentDoc + self.DocInBatch, self.D_All)
        self.D = self.currentDoc - self.currentDoc
        print("\t" + str(self.D) + " documents will be analyze")
        for d in range(self.startDoc, self.currentDoc):
            document = documentSet.documents[d]
            documentID = document.documentID
            cluster, _, __ = self.sampleCluster(d, document)  # Get initial cluster of each document
            self.z[documentID] = cluster
            if cluster not in self.m_z:
                self.m_z[cluster] = 0
            self.m_z[cluster] += 1
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                if cluster not in self.n_zv:
                    self.n_zv[cluster] = {}
                if wordNo not in self.n_zv[cluster]:
                    self.n_zv[cluster][wordNo] = 0
                self.n_zv[cluster][wordNo] += wordFre
                if cluster not in self.n_z:
                    self.n_z[cluster] = 0
                self.n_z[cluster] += wordFre
        self.batchNum += 1


        '''
            if documentID <= self.batchNum2tweetID[self.batchNum]:
                cluster, _, __ = self.sampleCluster(d, document)  # Get initial cluster of each document
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_z[cluster] += wordFre
                if d == self.D_All - 1:
                    self.startDoc = self.currentDoc
                    self.currentDoc = self.D_All
                    self.batchNum += 1
            else:
                self.startDoc = self.currentDoc
                self.currentDoc = d
                self.batchNum += 1
                break
        '''

    def gibbsSampling(self, documentSet):
        self.pro_cal = [0.0, 0.0]
        for i in range(self.iterNum):
            print("\titer is ", i)
            for d in range(self.startDoc, self.currentDoc):
                document = documentSet.documents[d]
                documentID = document.documentID
                cluster = self.z[documentID]
                self.m_z[cluster] -= 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre
                    self.n_z[cluster] -= wordFre
                self.checkEmpty(cluster)
                (cluster, pro_max, pro_lmax) = self.sampleCluster(d, document)
                # cluster = self.sampleCluster(d, document)
                if i == self.iterNum - 1:
                    self.pro_cal[0] += pro_max
                    self.pro_cal[1] += pro_lmax
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    self.n_z[cluster] += wordFre
        self.pro_cal[0] = self.pro_cal[0]/(self.currentDoc - self.startDoc)
        self.pro_cal[1] = self.pro_cal[1]/(self.currentDoc - self.startDoc)


    def sampleCluster(self, d, document):
        prob = [float(0.0)] * (self.K + 1)
        for cluster in range(self.K):
            if cluster not in self.m_z or self.m_z[cluster] == 0:
                self.m_z[cluster] = 0
                prob[cluster] = 0
                continue
            prob[cluster] = self.m_z[cluster] / (self.D - 1 + self.alpha)
            valueOfRule2 = 1.0
            i = 0
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                for j in range(wordFre):
                    if cluster not in self.n_zv:
                        self.n_zv = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    valueOfRule2 *= (self.n_zv[cluster][wordNo] + self.beta + j) / (self.n_z[cluster] + self.beta0 + i)
                    i += 1
            prob[cluster] *= valueOfRule2
        prob[self.K] = self.alpha / (self.D - 1 + self.alpha)
        valueOfRule2 = 1.0
        i = 0
        for w in range(document.wordNum):
            wordFre = document.wordFreArray[w]
            for j in range(wordFre):
                valueOfRule2 *= (self.beta + j) / (self.beta0 + i)
                i += 1
        prob[self.K] *= valueOfRule2
        # 为什么这个样子？
        #regularization
        pro_sum = 0.0
        pro_max = 0.0
        pro_lmax = 0.0
        for k in range(0, self.K + 1):
            pro_sum += prob[k]
        for k in range(0, self.K + 1):
            prob[k] = prob[k] / pro_sum
            if pro_max < prob[k]:
                pro_lmax = pro_max
                pro_max = prob[k]
            elif pro_lmax < prob[k]:
                pro_lmax = prob[k]

        for k in range(1, self.K+1):
            prob[k] += prob[k-1]
        thred = random.random() * prob[self.K]

        kChoosed = 0
        while kChoosed < self.K+1:
            if thred < prob[kChoosed]:
                break
            kChoosed += 1
        if kChoosed == self.K:
            self.K += 1
            self.K_current += 1
        # return kChoosed
        return kChoosed, pro_max, pro_lmax

    def checkEmpty(self, cluster):
        if self.m_z[cluster] == 0:
            self.K_current -= 1

    def output(self, documentSet, outputPath, wordList, batchNum):
        outputDir = outputPath + self.dataset + self.ParametersStr + "Batch" + str(batchNum) + "/"
        try:
            isExists = os.path.exists(outputDir)
            if not isExists:
                os.mkdir(outputDir)
                print("\tCreate directory:", outputDir)
        except:
            print("ERROR: Failed to create directory:", outputDir)
        self.outputClusteringResult(outputDir, documentSet)
        self.estimatePosterior()
        self.outputPhiWordsInTopics(outputDir, wordList, self.wordsInTopicNum)
        self.outputSizeOfEachCluster(outputDir, documentSet)

    def estimatePosterior(self):
        self.phi_zv = {}
        for cluster in self.n_zv:
            n_z_sum = 0
            if self.m_z[cluster] != 0:
                if cluster not in self.phi_zv:
                    self.phi_zv[cluster] = {}
                for v in self.n_zv[cluster]:
                    if self.n_zv[cluster][v] != 0:
                        n_z_sum += self.n_zv[cluster][v]
                for v in self.n_zv[cluster]:
                    if self.n_zv[cluster][v] != 0:
                        self.phi_zv[cluster][v] = float(self.n_zv[cluster][v] + self.beta) / float(n_z_sum + self.beta0)

    def getTop(self, array, rankList, Cnt):
        index = 0
        m = 0
        while m < Cnt and m < len(array):
            max = 0
            for no in array:
                if (array[no] > max and no not in rankList):
                    index = no
                    max = array[no]
            rankList.append(index)
            m += 1

    def outputPhiWordsInTopics(self, outputDir, wordList, Cnt):
        outputfiledir = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "PhiWordsInTopics.txt"
        writer = open(outputfiledir, 'w')
        for k in range(self.K):
            rankList = []
            if k not in self.phi_zv:
                continue
            topicline = "Topic " + str(k) + ":\n"
            writer.write(topicline)
            self.getTop(self.phi_zv[k], rankList, Cnt)
            for i in range(rankList.__len__()):
                tmp = "\t" + wordList[rankList[i]] + "\t" + str(self.phi_zv[k][rankList[i]])
                writer.write(tmp + "\n")
        writer.close()

    def outputSizeOfEachCluster(self, outputDir, documentSet):
        outputfile = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "SizeOfEachCluster.txt"
        writer = open(outputfile, 'w')
        topicCountIntList = []
        for cluster in range(self.K):
            if self.m_z[cluster] != 0:
                topicCountIntList.append([cluster, self.m_z[cluster]])
        line = ""
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n\n")
        line = ""
        topicCountIntList.sort(key = lambda tc: tc[1], reverse = True)
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n")
        writer.close()

    def outputClusteringResult(self, outputDir, documentSet):
        outputPath = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "ClusteringResult" + ".txt"
        writer = open(outputPath, 'w')
        for d in range(self.startDoc, self.currentDoc):
            documentID = documentSet.documents[d].documentID
            cluster = self.z[documentID]
            writer.write(str(documentID) + " " + str(cluster) + "\n")
        writer.close()

    def save_pro_iter(self, batchNum, iterNum, pro_value):
        pro_path = 'pro_result'
        try:
            isExists = os.path.exists(pro_path)
            if not isExists:
                os.mkdir(pro_path)
                print("\tCreate directory:", pro_path)
        except:
            print("ERROR: Failed to create directory:", pro_path)
        iter_path = pro_path + '/iter' + str(iterNum)
        try:
            isExists = os.path.exists(iter_path)
            if not isExists:
                os.mkdir(iter_path)
                print("\tCreate directory:", iter_path)
        except:
            print("ERROR: Failed to create directory:", pro_path)
        if batchNum == 1:
            writer = open(iter_path + '/result.txt', 'w')
        else:
            writer = open(iter_path + '/result.txt', 'a')
        writer.write(str(pro_value[0]) + ' ' + str(pro_value[1]) + '\n')
        writer.close()
