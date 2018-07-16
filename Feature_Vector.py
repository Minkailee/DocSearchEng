# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:46:57 2018

@author: Lenovo
"""

import os
import re
from bs4 import BeautifulSoup
import nltk
#import math
import string
import numpy
from scipy import sparse
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import *

from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib

def PreprocessFile(filename, fmt):
    if fmt ==  "none":
        s = filename
    elif fmt != "txt":      
        soup = BeautifulSoup(open(filename, encoding="ISO-8859-1"), "lxml")
        s = str(soup.body)
        s = re.sub(r'[\xbd\xc9\xba\xd6\xef\xe2\xeb\xb4\xa2\xb3\xa3\xb8\xbc\xae\xbe\xf6\xb2\u2022\xc2\xa9\xf1\xe7\xf8\xd8\xa5\xad\xdf\xcd\xe4\xf4\u20ac\xc7\xc8\xdc\xb9\xa6\xe6\u2154\xee\u02dc\u2153\u2122\u25aa\u24c7\xaf\xc0\u20a4\u2713\uf07f\uf073\u02c6\u015b\u0109\uf02d\uf0b8\u2028\xc3\uf0e0\uf0ae\u2003\uf0d8\u04b0\uf097\uf05c\uf02a\u2212\uf09f\xf5\uf054\uf0fc\u200b\uf03e\xb5\ufb00\u25ca\ufb03\u037e\u2126\u0336\u2794\u0107\ufffd\u2250\uf0a7\u201e\u03cb\u2020\u0131\u02da\u0334\u25cc\u0627\xbb\u21d2\u0387\u0153\u2206\u201b\u24c1\x9d\uf0b0\uf020\xe3\u25e6\u0644\u0633]','',s)
        try:
            start = s.index("<!--sino index-->")
            stop = s[start:].index("<!--sino noindex-->")
            s = s[start:start + stop]
        except ValueError as error:
            print(error)
    else:
        with open(filename) as f:
            s = f.read()
    s = re.sub(r'<a.*/a>', ' ', s)
    s = re.sub(r'<[^<>]*>', ' ', s)
    s = re.sub(r'&\w+', ' ', s)
    s = re.sub(r'[^\ws]',' ',s).lower()
    rule = re.compile(r'[^a-zA-z]')
    s = rule.sub(' ',s)
    data = s.split()
    results = ""
    stopWords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    for word in data:
        if len(word)>1 and word not in stopWords:
            results += (stemmer.stem(word) + " ")
    print(results)
    return results

def count_term(filename, ForW):
    if ForW == "f":
        temp = filename.split(".")
        fmt = temp[1]
    else:
        fmt = "none"
    text = PreprocessFile(filename, fmt)
    tokens = nltk.word_tokenize(text)
    count = Counter(tokens)
    return count


def LSA_Compress(Vector, targetDimension):
    estimator = TruncatedSVD(n_components=targetDimension)
    ca_Vector = estimator.fit_transform(Vector)
    return ca_Vector, estimator

def get_WORDSET(fileDir): 
    filelist = []
    words_set = []
    file = open("D:/AU_STUDY/9900/words_set",'w+')
    file.truncate()
    trace_mark = 0
    for parent,dirnames,filenames in os.walk(fileDir):
        for filename in filenames:
            filelist.append(filename)
#            print(filename)
            with open("result/" + filename) as f:
                trace_mark += 1
                if trace_mark%1000 == 0:
                    print(trace_mark)
                count = 0
                lines = f.readlines()
                for line in lines:
                    if count<20:
                        l=line.strip().split()
                        if l[0] not in words_set:
                            words_set.append(l[0])
                            file.write(str((l[0])+'\n'))
                        count += 1
                    else:
                        break   
    file.close()
    return words_set


def get_WholeVector(countlist, words_set):
    row = []
    col = []
    data = []
    file_num = len(countlist)
    word_num = len(words_set)
    for i in range(0,file_num):
        counts = countlist[i]
        length = sum(counts.values())
        for j in range(0,word_num):
            if words_set[j] in counts:
                data.append(float(counts[words_set[j]]) / float(length))
                row.append(i)
                col.append(j)
    if data == []:
        return []
    data.append(0)
    row.append(file_num)
    col.append(word_num)
    rcd_shape = numpy.array([data, row, col])
    return rcd_shape

def get_CompressedFileInputVector(filename,words_set,estimator):
    countlist = [count_term(filename, "f")]
    rcd = get_WholeVector(countlist, words_set)
    if rcd == []:
        return []
    Vector = sparse.csr_matrix((rcd[0][:-1], (rcd[1][:-1], rcd[2][:-1])), shape=(int(rcd[1][-1]), int(rcd[2][-1])))      
    CompressedTestInputVector = estimator.transform(Vector)
    return numpy.array(CompressedTestInputVector)

def get_CompressedWordInputVector(words, words_set, estimator):       
    countlist = [count_term(words, "w")]
    rcd = get_WholeVector(countlist, words_set)
    if rcd == []:
        return []
    Vector = sparse.csr_matrix((rcd[0][:-1], (rcd[1][:-1], rcd[2][:-1])), shape=(int(rcd[1][-1]), int(rcd[2][-1])))      
    CompressedTestInputVector = estimator.transform(Vector)
    return numpy.array(CompressedTestInputVector)
    

def text_read(filename):
    try:
        file = open(filename,'r')
    except IOError:
        error = []
        return error
    content = file.readlines()

    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]

    file.close()
    return content

