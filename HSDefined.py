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

def load_HS_corpus(maxlength, window_size):
    trigram2id={}
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
    
    allowed_read_sentence=0
    def store_into_matrices(sent, contexts, targets, targetWords_per_sentence, allowed_read_sentence):
        if len(sent)==0:
            return allowed_read_sentence
        else:
            data.append(sent)
            #print 'sent'
            #print sent
            #print 'targets'
            #print targets
            repeated_targets=targets*60
            target_matrix.append(repeated_targets[:60])  #only consider maxmum 60 target words for training
            #print 'context'
            #print contexts
            repeated_context=contexts*60
            context_matrix.append(repeated_context[:(60*window_size*2)]) # consider 60*(context_size)
            Lengths_target.append(targetWords_per_sentence)
            #print 'Lengths_target'
            #print Lengths_target
            allowed_read_sentence+=1
            #print 'already '+str(allowed_read_sentence)+' sentences loaded..'
            
            return allowed_read_sentence
            
    sent=[]
    contexts=[]
    targets=[]
    targetWords_per_sentence=0
    valid_sent=True
    readFile=open('/mounts/Users/cisintern/hs/l/schuetze2014/yin/cnnlm/wiki,ebert,uniq,ngram,small,input,to,cnnlm.txt')
    for line in readFile:
        strip_line=line.strip()
        pos=strip_line.find(':')
        label=strip_line[:pos]
        tokens=strip_line[pos+1:].strip()
        if label=='SENTENCE':
            valid_sent=True
            #print 'loading SENTENCE'
            allowed_read_sentence=store_into_matrices(sent, contexts, targets, targetWords_per_sentence, allowed_read_sentence)

            sent=[]
            contexts=[]
            targets=[]
            
            #if allowed_read_sentence==6:
            #    break
            targetWords_per_sentence=0 # clear this variable, to store new words of this sentence
            trigrams=tokens.split()
            length=len(trigrams)
            Lengths.append(length)
            left=(maxlength-length)/2
            right=maxlength-left-length
            leftPad.append(left)
            rightPad.append(right)
            if left<0 or right<0:
                #print 'Too long sentence:\n'+tokens
                valid_sent=False
                Lengths.pop()
                leftPad.pop()
                rightPad.pop()
                continue    #read next line
            
            
            sent+=[0]*left
            for trigram in trigrams:
                #sent.append(word2id.get(word))
                    
                id=trigram2id.get(trigram, -1)   # possibly the embeddings are for words with lowercase
                if id == -1:
                    #embeddings.append(numpy.random.uniform(-1,1,embedding_size)) # generate a random embedding for an unknown word
                    #embeddings_target.append(numpy.random.uniform(-1,1,embedding_size))
                    trigram2id[trigram]=trigram_count+1   # starts from 1
                    #id2trigram[trigram_count]=1 #1 means new words
                    sent.append(trigram_count+1)
                    trigram_count+=1                  
                else:
                    sent.append(id)
            sent+=[0]*right
            #data.append(sent)

        elif label=='LEFT CONTEXT' and valid_sent:
            #print 'loading LEFT_CONTEXT: '+tokens
            
            left_contexts=tokens.split()
            #print 'length of left contexts:'+str(len(left_contexts))

            if window_size != len(left_contexts):
                print 'context length != window_size: '+str(window_size)
                exit(0)
            for context in left_contexts:
                #print context
                id=trigram2id.get(context,0)   #padd is set to index 0
                #print id
                contexts.append(id)
        elif label=='TARGET WORD' and valid_sent:
            #print 'loading TARGET WORD: '+tokens
            all_word_numbers+=1
            targetWords_per_sentence+=1
            word=tokens.strip()
            id=target_word2id.get(word,-1)
            if id==-1:  #a new word
                target_word2id[word]=word_count # word index starts from 0
                target_id2word[word_count]=word
                id=word_count
                wordid2count[id]=1 #the first time to appear
                word_count+=1
            else:
                wordid2count[id]=wordid2count[id]+1
            targets.append(id)
            
            
        elif label=='RIGHT CONTEXT' and valid_sent:
            #print 'loading RIGHT CONTEXT: '+tokens
            right_contexts=tokens.split()
            if window_size != len(right_contexts):
                print 'context length != window_size: '+str(window_size)
                exit(0)
            for context in right_contexts:
                id=trigram2id.get(context,0)
                contexts.append(id)                  
    allowed_read_sentence=store_into_matrices(sent, contexts, targets, targetWords_per_sentence, allowed_read_sentence) #the last sentence
    print 'HS corpus loaded over.'
    unigram=[]
    for index in xrange(word_count):
        unigram.append(wordid2count[index]*1.0/all_word_numbers)  
        #unigram.append(1.0/all_word_numbers)  
        
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
    if len(data)!=len(leftPad) or len(data)!=len(Lengths) or len(data)!=len(Lengths_target) or len(data)!=len(context_matrix) or len(data)!=len(target_matrix):
        print 'Load data error: sentence amount not equal to padding.'
        exit(0)
    rval = [(numpy.array(data),train_set_Lengths, train_left_pad, train_right_pad), (numpy.array(data),train_set_Lengths, train_left_pad, train_right_pad)]
    return rval, numpy.array(unigram), numpy.array(Lengths), numpy.array(Lengths_target),trigram_count, numpy.array(context_matrix), numpy.array(target_matrix), target_id2word
        
    #return numpy.array(data),context_matrix, target_matrix, numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad), trigram_count, trigram2id, numpy.array(unigram)
        