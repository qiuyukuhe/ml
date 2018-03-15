#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @time    : 2018/3/13 22:01
# @author  : wangxinxi
# @file    : bayes.py
# @aoftware: PyCharm

from numpy import *
from functools import reduce

#广告、垃圾标识
adClass = 1

def loadDataSet() :
	'''加载数据集合及其对应的分类'''
	wordsList = [['周六', '公司', '一起', '聚餐','时间'],
	             ['优惠', '返利', '打折', '优惠', '金融', '理财'],
	             ['喜欢', '机器学习', '一起', '研究', '欢迎', '贝叶斯', '算法', '公式'],
	             ['公司', '发票', '税点', '优惠', '增值税', '打折'],
	             ['北京', '今天', '雾霾', '不宜', '外出', '时间', '在家', '讨论', '学习'],
	             ['招聘', '兼职', '日薪', '保险', '返利']]
	# 1 是, 0 否
	classVec = [0, 1, 0, 1, 0, 1]
	return wordsList, classVec


def doc2VecList(docList) :
	'''合并去重，生成包含所有单词的集合'''
	return list(reduce(lambda x, y : set(x) | set(y), docList))


def words2Vec(vecList, inputWords) :
	'''把单子转化为词向量'''
	resultVec = [0] * len(vecList)
	for word in inputWords :
		if word in vecList :
			# 在单词出现的位置上的计数加1
			resultVec[vecList.index(word)] += 1
		else :
			print('没有发现此单词')
	
	return array(resultVec)


def trainNB(trainMatrix, trainClass) :
	'''计算，生成每个词对于类别上的概率'''
	numTrainClass = len(trainClass)
	numWords = len(trainMatrix[0])
	
	# 全部都初始化为1， 防止出现概率为0的情况出现
	p0Num = ones(numWords)
	p1Num = ones(numWords)
	# 相应的单词初始化为2
	p0Words = 2.0
	p1Words = 2.0
	# 统计每个分类的词的总数
	for i in range(numTrainClass) :
		if trainClass[i] == 1 :
			#数组在对应的位置上相加
			p1Num += trainMatrix[i]
			p1Words += sum(trainMatrix[i])
		else :
			p0Num += trainMatrix[i]
			p0Words += sum(trainMatrix[i])
	# 计算每种类型里面， 每个单词出现的概率
	p0Vec = log(p0Num / p0Words)
	p1Vec = log(p1Num / p1Words)
	# 计算1出现的概率
	pClass1 = sum(trainClass) / float(numTrainClass)
	return p0Vec, p1Vec, pClass1


def classifyNB(testVec, p0Vec, p1Vec, pClass1) :
	'''朴素贝叶斯分类, max(p0， p1)作为推断的分类'''
	# y=x 是单调递增的， y=ln(x)也是单调递增的。 ， 如果x1 > x2, 那么ln(x1) > ln(x2)
	# 因为概率的值太小了，所以我们可以取ln， 根据对数特性ln(ab) = lna + lnb， 可以简化计算
	# sum是numpy的函数
	p1 = sum(testVec * p1Vec) + log(pClass1)
	p0 = sum(testVec * p0Vec) + log(1 - pClass1)
	if p0 > p1 :
		return 0
	return 1

def printClass(words, testClass):
	if testClass == adClass:
		print(words, '推测为：广告邮件')
	else:
		print(words, '推测为：正常邮件')

def tNB() :
	'''测试，进行预测'''
	docList, classVec = loadDataSet()
	# 生成包含所有单词的list
	allWordsVec = doc2VecList(docList)
	
	# 构建词向量矩阵
	trainMat = list(map(lambda x : words2Vec(allWordsVec, x), docList))
	# 训练计算每个词在分类上的概率, p0V:每个单词在非分类出现的概率， p1V:每个单词在是分类出现的概率
	p0V, p1V, pClass1 = trainNB(trainMat, classVec)
	testWords = ['公司', '聚餐', '讨论', '贝叶斯']
	# 当前需要预测的词向量
	testVec = words2Vec(allWordsVec, testWords)
	# 预测分类
	testClass = classifyNB(testVec, p0V, p1V, pClass1)
	printClass(testWords, testClass)
	
	# 预测
	testWords = ['公司', '保险', '金融']
	testVec = words2Vec(allWordsVec, testWords)
	testClass = classifyNB(testVec, p0V, p1V, pClass1)
	printClass(testWords, testClass)


if __name__ == '__main__' :
	tNB()
