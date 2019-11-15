# -*- coding: utf-8 -*-
#! /usr/bin/python
from __future__ import print_function
from multiprocessing import Process
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import Confusion_Show as cs
import LoadSolitData
from sklearn.model_selection import GridSearchCV



def write2txt(data,storename):
    f = open(storename,"w")
    f.write(str(data))
    f.close()

def V_candidate(candidatelist,train_x, train_y):
    model = RandomForestClassifier()
    param_grid = {'n_estimators': [i for i in candidatelist]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1,cv=10)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    return best_parameters["n_estimators"]


def random_forest_classifier(train_x, train_y,minedge,maxedge,step):

    model = RandomForestClassifier()
    param_grid = {'n_estimators': [i for i in range(minedge,maxedge,step)]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1,cv=10)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    return best_parameters["n_estimators"]


def FlexSearch(train_x, train_y,minedge,maxedge,step,storename):
    mededge = random_forest_classifier(train_x, train_y,minedge,maxedge,step)
    minedge = mededge - step-step/2
    maxedge = mededge + step+step/2
    step = step/2
    print("Current bestPara:",mededge)
    print("Next Range%d-%d"%(minedge,maxedge))
    print("Next step:",step)
    if step > 0:
        return FlexSearch(train_x, train_y,minedge,maxedge,step,storename)
    elif step==0:
        print("The bestPara:",mededge)
        write2txt(mededge,"RF_%s.txt"%storename)


def Calc(classifier):
    Classifier.fit(train_data,train_label)
    predict_label = Classifier.predict(test_data)
    predict_label_prob = Classifier.predict_proba(test_data)

    total_cor_num = 0.0
    dictTotalLabel = {}
    dictCorrectLabel = {}
    for label_i in range(len(predict_label)):

        if predict_label[label_i] == test_label[label_i]:
            total_cor_num += 1
            if predict_label[label_i] not in dictCorrectLabel: dictCorrectLabel[predict_label[label_i]] = 0
            dictCorrectLabel[predict_label[label_i]] += 1.0
        if test_label[label_i] not in dictTotalLabel: dictTotalLabel[test_label[label_i]] = 0
        dictTotalLabel[test_label[label_i]] += 1.0

    accuracy = metrics.accuracy_score(test_label, predict_label) * 100
    kappa_score = metrics.cohen_kappa_score(test_label, predict_label)
    average_accuracy = 0.0
    label_num = 0

    for key_i in dictTotalLabel:
        try:
            average_accuracy += (dictCorrectLabel[key_i] / dictTotalLabel[key_i]) * 100
            label_num += 1
        except:
            average_accuracy = average_accuracy

    average_accuracy = average_accuracy / label_num

    result = "OA:%.4f;AA:%.4f;KAPPA:%.4f"%(accuracy,average_accuracy,kappa_score)
    print(result)
    report = metrics.classification_report(test_label, predict_label)
    print(report)
    cm = metrics.confusion_matrix(test_label, predict_label)
    cs.ConfusionMatrixPng(cm,['1','2','3','4','5','6','7','8','9','10','11','12','13'])


if __name__ == '__main__':

    train_data,train_label,test_data,test_label = LoadSolitData.TrainSize(1,0.5,[i for i in range(1,14)],"outclass") # 数据集自导自己的

    minedge = 180
    maxedge = 330
    step = 10
    gap = (maxedge-minedge)/3
    subprocess = []
    for i in range(1,4):#设置了3个进程
        p = Process(target=FlexSearch,args=(train_data,train_label,minedge+(i-1)*gap,minedge+i*gap,step,i))
        subprocess.append(p)
    for j in subprocess:
        j.start()
    for k in subprocess:
        k.join()

    candidatelist = []
    for i in range(1,4):
        with open("RF_%s.txt"%i) as f:
            candidatelist.append(int(f.readlines()[0].strip()))

    print("candidatelist:",candidatelist)

    best_n_estimators = V_candidate(candidatelist, train_data,train_label)
    print("best_n_estimators",best_n_estimators)
    Classifier = RandomForestClassifier(n_estimators=best_n_estimators)
    Calc(Classifier)