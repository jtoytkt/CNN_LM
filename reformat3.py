import string                 
import gzip
import os
import sys
import time

import numpy

import theano.sandbox.neighbours as TSN
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from cis.deep.utils.theano import debug_print                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                               
numcontext = 10                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                               
class trainex:                                                                                                                                                                                                                                                                 
   def triple(self):   
      #print (self.word,self.left,self.rght)                                                                                                                                                                                                                                                      
      return (self.word,self.left,self.rght)          
      #print self.word, self.left, self.rght                                                                                                                                                                                                                          
   def __init__(self,mywords,i,sent,i2w):                                                                                                                                                                                                                                      
      self.word = i2w[mywords[i]]                                                                                                                                                                                                                                              
      self.left = []                                                                                                                                                                                                                                                           
      j = i                                                                                                                                                                                                                                                                    
      while len(self.left)<numcontext:                                                                                                                                                                                                                                         
         j -= 1                                                                                                                                                                                                                                                                
         if j<0:                                                                                                                                                                                                                                                               
            self.left.append('padd')                                                                                                                                                                                                                                           
         elif mywords[j]==mywords[i]:                                                                                                                                                                                                                                          
            continue                                                                                                                                                                                                                                                           
         else:                                                                                                                                                                                                                                                                 
            self.left.append(sent[j])                                                                                                                                                                                                                                          
      self.left.reverse()                                                                                                                                                                                                                                                      
      self.rght = []                                                                                                                                                                                                                                                           
      j = i                                                                                                                                                                                                                                                                    
      while len(self.rght)<numcontext:                                                                                                                                                                                                                                         
         j += 1                                                                                                                                                                                                                                                                
         if j>=len(sent):                                                                                                                                                                                                                                                      
            self.rght.append('padd')                                                                                                                                                                                                                                           
         elif mywords[j]==mywords[i]:                                                                                                                                                                                                                                          
            continue                                                                                                                                                                                                                                                           
         else:                                                                                                                                                                                                                                                                 
            self.rght.append(sent[j])

class wikiline:
   def sentence(self):
      return self.sent
   def triples(self):
      for myex in self.alltrainex:
         myex.triple()
   def __init__(self,myline):
      myparts = string.split(myline,'5')
      self.sent = []
      self.alltrainex = []
      mywords = []
      i2w = {}
      for i,myword in enumerate(myparts):
         if myword=='': continue
         subparts = string.split(myword,'6')
         #print subparts
         assert len(subparts)==2
         i2w[i] = string.strip(subparts[0])
         subsubparts = string.split(subparts[1])
         for subsubpart in subsubparts:
            if subsubpart=='': continue
            self.sent.append(subsubpart)
            mywords.append(i)
      done = set()
      for i in range(len(self.sent)):
         if mywords[i] in done: continue
         done.add(mywords[i])
         self.alltrainex.append(trainex(mywords,i,self.sent,i2w))

def yinwikireformat3(maxlength, window_size, max_size):
    #defined by wenpeng
    trigram2id={}
    id2trigram={}
    target_word2id={}
    target_id2word={}
    wordid2count={}
    trigram_count=0
    word_count=0
    all_word_numbers=0
    data=[]
    context_matrix=[]
    target_matrix=[]
    Lengths=[]
    Lengths_target=[]
    leftPad=[]
    rightPad=[]
    filename = '/mounts/Users/cisintern/hs/l/schuetze2014/yin/cnnlm/wiki,ebert,uniq,ngram.txt'
    myfile = open(filename,'r')
    count=0
    for myline in myfile:
        #print myline
        myline = string.strip(myline)
        if myline=='': continue
        myobj = wikiline(myline)
        '''
        print
        print myobj.sentence()
        #myobj.triples()
        '''

        
        input_length=len(myobj.sentence())
        #print 'input_length: '+str(input_length)
        if input_length>maxlength or input_length<20:
            continue
        else: # a valid sentence
            if count==max_size:
                break
            count+=1
            sent=[]
            contexts=[]
            targets=[]
            targetWords_per_sentence=0
            
            Lengths.append(input_length)
            left=(maxlength-input_length)/2
            right=maxlength-left-input_length
            leftPad.append(left)
            rightPad.append(right)        
            sent+=[0]*left
            for trigram in myobj.sentence():
                #print trigram
                    
                id=trigram2id.get(trigram, -1)   # possibly the embeddings are for words with lowercase
                if id == -1:
                    #embeddings.append(numpy.random.uniform(-1,1,embedding_size)) # generate a random embedding for an unknown word
                    #embeddings_target.append(numpy.random.uniform(-1,1,embedding_size))
                    trigram2id[trigram]=trigram_count+1   # starts from 1
                    id2trigram[trigram_count+1]=trigram #1 means new words
                    sent.append(trigram_count+1)
                    trigram_count+=1                  
                else:
                    sent.append(id)
            sent+=[0]*right    
            #left context
            for myex in myobj.alltrainex:
                target_word, left_contexts, right_contexts=myex.triple()
                #print target_word, left_contexts, right_contexts
                targetWords_per_sentence+=1
                #store context for target word
                for context in left_contexts:
                    #print context
                    id=trigram2id.get(context,0)   #padd is set to index 0
                    contexts.append(id)
                    if id==0:
                        id2trigram[id]=context
                for context in right_contexts:
                    id=trigram2id.get(context,0)   #padd is set to index 0
                    contexts.append(id)   
                    if id==0:
                        id2trigram[id]=context
                #store target word
                all_word_numbers+=1
                id=target_word2id.get(target_word,-1)
                if id==-1:  #a new word
                    target_word2id[target_word]=word_count # word index starts from 0
                    target_id2word[word_count]=target_word
                    id=word_count
                    wordid2count[id]=1 #the first time to appear
                    word_count+=1
                else:
                    wordid2count[id]=wordid2count[id]+1
                targets.append(id)    

            data.append(sent)
            repeated_targets=targets*60
            target_matrix.append(repeated_targets[:60])  #only consider maxmum 60 target words for training
            repeated_context=contexts*60
            context_matrix.append(repeated_context[:(60*window_size*2)]) # consider 60*(context_size)
            Lengths_target.append(targetWords_per_sentence)           


    '''
    print 'data'
    print data
    print 'target_matrix'
    print target_matrix
    print 'context_matrix'
    print context_matrix        
    print 'Lengths_target'                  
    print Lengths_target
    '''
    print 'Wiki corpus loaded over. Totally '+str(word_count)+' distinct target words'
    unigram=[]
    for index in xrange(word_count):
        unigram.append(wordid2count[index]*1.0/all_word_numbers)        
    #print unigram 
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int32')
        #return shared_y

    train_set_Lengths=debug_print(shared_dataset(numpy.array(Lengths)),'train_set_length' )                            
    #valid_set_Lengths = shared_dataset(devLengths)
    #uni_gram=shared_dataset(unigram)
    
    train_left_pad=debug_print(shared_dataset(numpy.array(leftPad)),'leftPad')
    train_right_pad=debug_print(shared_dataset(numpy.array(rightPad)), 'rightPad')
    #dev_left_pad=shared_dataset(devLeftPad)
    #dev_right_pad=shared_dataset(devRightPad)
    '''
    print 'length_target:'
    print Lengths_target
    print 'context_matrix:'
    print context_matrix
    print 'target_matrix:'
    print target_matrix
    '''
    if trigram_count+1!=len(id2trigram):
        print 'trigram_count: '+str(trigram_count)
        print 'id2trigram:'+str(len(id2trigram))
        exit(0)
    if len(data)!=len(leftPad) or len(data)!=len(Lengths) or len(data)!=len(Lengths_target) or len(data)!=len(context_matrix) or len(data)!=len(target_matrix):
        print 'Load data error: sentence amount not equal to padding.'
        exit(0)
    rval = [(numpy.array(data),train_set_Lengths, train_left_pad, train_right_pad), (numpy.array(data),train_set_Lengths, train_left_pad, train_right_pad)]
    return rval, numpy.array(unigram), numpy.array(Lengths), numpy.array(Lengths_target),trigram_count, numpy.array(context_matrix), numpy.array(target_matrix), target_id2word, id2trigram
 
      
#yinwikireformat3(250, 10)
