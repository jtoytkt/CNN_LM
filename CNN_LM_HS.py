
import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy
import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from WPDefined import ConvFoldPoolLayer,Conv_Fold_DynamicK_PoolLayer, dropout_from_layer, shared_dataset, load_model_for_training, SoftMaxlayer
from word2embeddings.nn.layers import BiasedHiddenLayer, SerializationLayer, \
    IndependendAttributesLoss, SquaredErrorLossLayer
from word2embeddings.nn.util import zero_value, random_value_normal, \
    random_value_GloBen10
from word2embeddings.tools.theano_extensions import MRG_RandomStreams2
from cis.deep.utils.theano import debug_print
from HSDefined import load_HS_corpus
from reformat3 import yinwikireformat3

class CNN_LM(object):
    def __init__(self, learning_rate=0.2, n_epochs=2000, nkerns=[6, 14], batch_size=10, useAllSamples=0, ktop=4, filter_size=[7,5],
                    L2_weight=0.00005, dropout_p=0.8, useEmb=0, task=2, corpus=1, dataMode=3, maxSentLength=600, sentEm_length=48, window=3, 
                    k=5, nce_seeds=2345, only_left_context=False, vali_cost_list_length=20, context_embedding_size=48, train_scheme=1, max_size=10):
        self.write_file_name_suffix='_nk'+str(nkerns[0])+'&'+str(nkerns[1])+'_bs'+str(batch_size)+'_fs'+str(filter_size[0])+'&'+str(filter_size[1])\
        +'_maxSL'+str(maxSentLength)+'_window'+str(window)+'_noise'+str(k)+'_wait'+str(vali_cost_list_length)+'_conEm'+str(context_embedding_size)\
        +'_maxS'+str(max_size)
        #print self.write_file_name_suffix
        #exit(0)
        
        self.ini_learning_rate=learning_rate
        self.n_epochs=n_epochs
        self.nkerns=nkerns
        self.batch_size=batch_size
        self.useAllSamples=useAllSamples
        
        self.ktop=ktop
        self.filter_size=filter_size
        self.L2_weight=L2_weight
        self.dropout_p=dropout_p
        self.useEmb=useEmb
        self.task=task
        self.corpus=corpus
        self.dataMode=dataMode
        self.maxSentLength=maxSentLength
        self.kmax=self.maxSentLength/2+5
        self.sentEm_length=sentEm_length
        self.window=window
        self.k=k
        self.only_left_context=only_left_context
        if self.only_left_context:
            self.context_size=self.window
        else:
            self.context_size=2*self.window
        self.nce_seed=nce_seeds
        self.context_embedding_size=context_embedding_size
        self.train_scheme=train_scheme
        '''
        root="/mounts/data/proj/wenpeng/Dataset/StanfordSentiment/stanfordSentimentTreebank/"
        wiki_path="/mounts/data/proj/wenpeng/PhraseEmbedding/enwiki-20130503-pages-articles-cleaned-tokenized"
        embeddingPath='/mounts/data/proj/wenpeng/Downloads/hlbl-embeddings-original.EMBEDDING_SIZE=50.txt'
        embeddingPath2='/mounts/data/proj/wenpeng/MC/src/released_embedding.txt'
        '''
        self.max_size=max_size
        datasets, unigram, train_lengths, target_lengths, trigram_count, context_matrix, target_matrix, target_id2word, id2trigram=yinwikireformat3(self.maxSentLength, self.window, self.max_size)
        #exit(0)

        
        
        self.datasets=datasets
        self.context_matrix=context_matrix
        self.target_matrix=target_matrix
        self.trigram_size=trigram_count
        #print 'trigram_size is: '+str(trigram_count)
        self.target_id2word=target_id2word
        self.id2trigram=id2trigram
        '''
        self.target_embedding_size=200                                  
        rand_values=random_value_normal((len(target_id2word), self.target_embedding_size), theano.config.floatX, numpy.random.RandomState(4321))
        #rand_values[0]=numpy.array(numpy.zeros(self.embedding_size))
        self.embeddings_Q=theano.shared(value=rand_values)  
        ''' 
        #self.embeddings_Q=self.load_glove(target_id2word) # target embedding matrix
        self.target_embedding_size=0
        embed_R, embed_Q=self.new_load_glove(target_id2word)
        self.embeddings_Q=theano.shared(value=embed_Q)
        print 'target_embedding_size: '+str(self.target_embedding_size)
        
        #print 'self.embeddings_Q:'
        #print self.embeddings_Q.get_value()
        #rand_values=random_value_normal((self.trigram_size+1, self.context_embedding_size), theano.config.floatX, numpy.random.RandomState(1234))
        embed_R[0]=numpy.array(numpy.zeros(self.context_embedding_size))
        self.embeddings_R=theano.shared(value=embed_R)  
        #print 'self.embeddings_R:'
        #print self.embeddings_R.get_value()
        
        
        self.unigram=unigram   # is still a np.array()
        #print 'unigram:'
        #print self.unigram
        self.p_n=theano.shared(value=self.unigram)
        self.train_lengths=train_lengths
        '''
        print 'train_lengths:'
        print train_lengths
        '''
        self.target_lengths=target_lengths
        '''
        print 'target_lengths:'
        print self.target_lengths
        '''
        b_values = zero_value((len(unigram),), dtype=theano.config.floatX)
        self.bias = theano.shared(value=b_values, name='bias')
        self.vali_cost_list_length=vali_cost_list_length


    def new_load_glove(self, target_id2word):
        
        word2embeddings={}
        read_file=open('/mounts/data/proj/wenpeng/Dataset/embeddings-scaled.EMBEDDING_SIZE=200.txt')
        for line in read_file:
            tokens=line.strip().split()
            self.target_embedding_size=len(tokens)-1
            embedding=[]
            for i in range(1, self.target_embedding_size+1):
                embedding.append(float(tokens[i]))
            word2embeddings[tokens[0]]=embedding
        words_number=len(target_id2word)
        print 'totally '+str(words_number)+' distinct target words'
        embedding_Q=random_value_normal((len(target_id2word), self.target_embedding_size), theano.config.floatX, numpy.random.RandomState(4321))
        #for i in range(100):
        #    embedding_Q[0][i]=0.6
        
        unknown_words=0
        for index in range(words_number):
            embed=word2embeddings.get(target_id2word[index], -1)
            embed_lowercase=word2embeddings.get(target_id2word[index].lower(), -1)
            if embed==-1 and embed_lowercase==-1: # a unknown word which has no embedding in glove
                embedding_Q[index]=numpy.array(numpy.random.rand(self.target_embedding_size))
                unknown_words+=1
                #print target_id2word[index]
           
            elif embed!=-1:
                embedding_Q[index]=numpy.array(embed)
            else:
                embedding_Q[index]=numpy.array(embed_lowercase)
            
        print 'Collobert embeddings loaded over, '+str(unknown_words)+' words find no embeddings.'
        
        #numpy.savetxt('matrix.txt', embedding_Q, delimiter=',')
        #exit(0)
        embedding_R=random_value_normal((self.trigram_size+1, self.context_embedding_size), theano.config.floatX, numpy.random.RandomState(1234))
        '''
        count=0
        for word, embedding in word2embeddings.iteritems():
            embedding_R[count]=numpy.array(embedding[:self.context_embedding_size])
            count+=1
            if count==(self.trigram_size+1):
                break
        '''
        return embedding_R, embedding_Q

   
    def load_glove(self, target_id2word):
        word2embeddings={}
        read_file=open('/mounts/data/proj/wenpeng/Dataset/embeddings-scaled.EMBEDDING_SIZE=200.txt')
        for line in read_file:
            tokens=line.strip().split()
            self.target_embedding_size=len(tokens)-1
            embedding=[]
            for i in range(1, self.target_embedding_size+1):
                embedding.append(float(tokens[i]))
            word2embeddings[tokens[0]]=embedding
        words_number=len(target_id2word)
        print 'totally '+str(words_number)+' distinct target words'
        #embedding_Q=random_value_normal((len(target_id2word), self.target_embedding_size), theano.config.floatX, numpy.random.RandomState(4321))
        embedding_Q=numpy.array(numpy.random.rand(len(target_id2word),self.target_embedding_size))
        '''
        unknown_words=0
        for index in range(words_number):
            embed=word2embeddings.get(target_id2word[index], -1)
            embed_lowercase=word2embeddings.get(target_id2word[index].lower(), -1)
            if embed==-1 and embed_lowercase==-1: # a unknown word which has no embedding in glove
                embedding_Q[index]=numpy.array(numpy.random.rand(self.target_embedding_size))
                unknown_words+=1
                print target_id2word[index]
           
            elif embed!=-1:
                embedding_Q[index]=numpy.array(embed)
            else:
                embedding_Q[index]=numpy.array(embed_lowercase)
            
        print 'glove embeddings loaded over, '+str(unknown_words)+' words find no embeddings.'
        #numpy.savetxt('matrix.txt', embedding_Q, delimiter=',')
        #exit(0)
        '''
        return embedding_Q
                
    
    def get_pure_noise(self, targets):
            # Create unigram noise distribution.
        srng = MRG_RandomStreams2(seed=self.nce_seed)
    
        # Get the indices of the noise samples.
        random_noise = srng.multinomial(size=(self.batch_size, self.k*4), pvals=self.unigram)
        noise_matrix=[]
        for row in range(self.batch_size):
            noise_list=[]
            target=targets[row][0]
            #print 'target:'+str(target)
            count=0
            for col in range(self.k*4):
                noise=debug_print(random_noise[row][col], 'noise')
                if noise.eval()!=target:
                    noise_list.append(noise)
                    count+=1
                    if count==self.k:
                        break
            noise_matrix.append(noise_list)
            
        random_noise=T.concatenate(noise_matrix, axis=0).reshape((self.batch_size, self.k))        
        noise_indices_flat = random_noise.reshape((self.batch_size * self.k,))
        p_n_noise = self.p_n[noise_indices_flat].reshape((self.batch_size, self.k))
        return random_noise, p_n_noise         
           
    def get_noise(self):
            # Create unigram noise distribution.
        srng = MRG_RandomStreams2(seed=self.nce_seed)
    
        # Get the indices of the noise samples.
        random_noise = srng.multinomial(size=(self.batch_size, self.k), pvals=self.unigram)
        #random_noise=theano.printing.Print('random_noise')(random_noise)
        noise_indices_flat = random_noise.reshape((self.batch_size * self.k,))
        p_n_noise = self.p_n[noise_indices_flat].reshape((self.batch_size, self.k))
        return random_noise, p_n_noise   
    
    def concatenate_sent_context(self,sent_matrix, context_matrix):
        return T.concatenate([sent_matrix, context_matrix], axis=1)
    
    def calc_r_h(self, h_indices):
        return self.embed_context(h_indices)
    
    def embed_context(self,indices):
        #indices is a matrix with (batch_size, context_size)
        embedded=self.embed_word_indices(indices, self.embeddings_R)
        '''
        flattened_embedded=embedded.flatten()
        batch_size=indices.shape[0]
        context_size=indices.shape[1]
        embedding_size=self.embeddings_R.shape[1]
        '''
        #we prefer concatenating context embeddings, it's different with Sebastian's code
        #return flattened_embedded.reshape((batch_size, context_size*embedding_size ))
        return embedded.reshape((self.batch_size, self.context_size*self.context_embedding_size))
    def embed_noise(self, indices):
        embedded=self.embed_word_indices(indices, self.embeddings_Q)
        '''
        flattened_embedded=embedded.flatten()
        return flattened_embedded.reshape((self.batch_size, self.k, self.embedding_size ))  
        '''
        return embedded.reshape((self.batch_size, self.k, self.target_embedding_size ))
    def embed_target(self,indices):
        embedded=self.embed_word_indices(indices, self.embeddings_Q)
        return embedded.reshape((self.batch_size, self.target_embedding_size ))       
    def embed_word_indices(self, indices, embeddings):
        indices2vector=indices.flatten()
        #return a matrix
        return embeddings[indices2vector]
    def extract_contexts_targets(self, indices_matrix, sentLengths, leftPad):
        #first pad indices_matrix with zero indices on both side
        left_padding = T.zeros((indices_matrix.shape[0], self.window), dtype=theano.config.floatX)
        right_padding = T.zeros((indices_matrix.shape[0], self.window), dtype=theano.config.floatX)
        matrix_padded = T.concatenate([left_padding, indices_matrix, right_padding], axis=1)  
        
        leftPad=leftPad+self.window   #a vector plus a number
           
        # x, y indices
        max_length=T.max(sentLengths)
        x=T.repeat(T.arange(self.batch_size), max_length)
        y=[]
        for row in range(self.batch_size):
            y.append(T.repeat((T.arange(leftPad[row], leftPad[row]+sentLengths[row]),), max_length, axis=0).flatten()[:max_length])
        y=T.concatenate(y, axis=0)   
        #construct xx, yy for context matrix
        context_x=T.repeat(T.arange(self.batch_size), max_length*self.context_size)
        #wenpeng=theano.printing.Print('context_x')(context_x)
        context_y=[]
        for i in range(self.window, 0, -1): # first consider left window
            context_y.append(y-i)
        if not self.only_left_context:
            for i in range(self.window): # first consider left window
                context_y.append(y+i+1)
        context_y_list=T.concatenate(context_y, axis=0)       
        new_shape = T.cast(T.join(0, 
                               T.as_tensor([self.context_size]),
                               T.as_tensor([self.batch_size*max_length])),
                               'int64')
        context_y_vector=T.reshape(context_y_list, new_shape, ndim=2).transpose().flatten()
        new_shape = T.cast(T.join(0, 
                               T.as_tensor([self.batch_size]),
                               T.as_tensor([self.context_size*max_length])),
                               'int64')
        
        context_matrix = T.reshape(matrix_padded[context_x,context_y_vector], new_shape, ndim=2)  
        new_shape = T.cast(T.join(0, 
                               T.as_tensor([self.batch_size]),
                               T.as_tensor([max_length])),
                               'int64') 
        target_matrix = T.reshape(matrix_padded[x,y], new_shape, ndim=2)
        return    T.cast(context_matrix, 'int64'),  T.cast(target_matrix, 'int64')
    def store_model_to_file(self):
        if self.train_scheme ==1:
            save_file = open('/mounts/data/proj/wenpeng/CNN_LM/model_params_HS'+self.write_file_name_suffix, 'wb')  # this will overwrite current contents
        elif self.train_scheme ==2 :
            save_file = open('/mounts/data/proj/wenpeng/CNN_LM/model_params_HS'+self.write_file_name_suffix, 'wb')  # this will overwrite current contents
        for para in self.best_params:           
            cPickle.dump(para.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
        save_file.close()
    
    def store_embeddings(self):
        embeddings_R=self.embeddings_R.get_value()
        embeddings_Q=self.embeddings_Q.get_value()
        if embeddings_R.shape[0]!=len(self.id2trigram) or embeddings_Q.shape[0]!=len(self.target_id2word):
            print 'embedding amount doesnt equal to id2tokens'
            print embeddings_R.shape[0], len(self.id2trigram), embeddings_Q.shape[0], len(self.target_id2word)
            exit(0)
        else:
            context_file=open('/mounts/data/proj/wenpeng/CNN_LM/context_embeddings_HS'+self.write_file_name_suffix+'.txt', 'w')
            for id in range(len(self.id2trigram)):
                context_file.write(self.id2trigram[id]+'\t')
                for j in range(self.context_embedding_size):
                    context_file.write(str(embeddings_R[id][j])+' ')
                context_file.write('\n')
            context_file.close()
            print 'context embedding stored over.'
            target_file=open('/mounts/data/proj/wenpeng/CNN_LM/target_embeddings_HS'+self.write_file_name_suffix+'.txt', 'w')
            for id in range(len(self.target_id2word)):
                target_file.write(self.target_id2word[id]+'\t')
                for j in range(self.target_embedding_size):
                    target_file.write(str(embeddings_Q[id][j])+' ')
                target_file.write('\n')
            target_file.close()
            print 'target embedding stored over.'
    def evaluate_lenet5(self):
    #def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, nkerns=[6, 12], batch_size=70, useAllSamples=0, kmax=30, ktop=5, filter_size=[10,7],
    #                    L2_weight=0.000005, dropout_p=0.5, useEmb=0, task=5, corpus=1):
        rng = numpy.random.RandomState(23455)
        
        #datasets, embedding_size, embeddings=read_data(root+'2classes/train.txt', root+'2classes/dev.txt', root+'2classes/test.txt', embeddingPath,60)

        #datasets = load_data(dataset)
        indices_train, trainLengths, trainLeftPad, trainRightPad= self.datasets[0]
        #indices_dev, devLengths, devLeftPad, devRightPad= self.datasets[1]
        '''
        print 'indices_train shapes:'
        print indices_train.shape[0], indices_train.shape[1]
        print indices_train
        '''
        #create embedding matrix to store the final embeddings
        sentences_embs=numpy.zeros((indices_train.shape[0],self.sentEm_length), dtype=theano.config.floatX)

        n_train_batches=indices_train.shape[0]/self.batch_size
        #n_valid_batches=indices_dev.shape[0]/self.batch_size
        remain_train=indices_train.shape[0]%self.batch_size
        
        train_batch_start=[]
        dev_batch_start=[]
        if self.useAllSamples:
            train_batch_start=list(numpy.arange(n_train_batches)*self.batch_size)+[indices_train.shape[0]-self.batch_size]
            #dev_batch_start=list(numpy.arange(n_valid_batches)*self.batch_size)+[indices_dev.shape[0]-self.batch_size]
            n_train_batches=n_train_batches+1
            #n_valid_batches=n_valid_batches+1
        else:
            train_batch_start=list(numpy.arange(n_train_batches)*self.batch_size)
            #dev_batch_start=list(numpy.arange(n_valid_batches)*self.batch_size)
        '''
        print 'train_batch_start:'
        print train_batch_start
        '''
        indices_train_theano=theano.shared(numpy.asarray(indices_train, dtype=theano.config.floatX), borrow=True)
        #indices_dev_theano=theano.shared(numpy.asarray(indices_dev, dtype=theano.config.floatX), borrow=True)
        indices_train_theano=T.cast(indices_train_theano, 'int32')
        '''
        print 'target_matrix shape'
        print self.target_matrix.shape[0], self.target_matrix.shape[1]
        print self.target_matrix
        '''
        indices_target_theano=theano.shared(numpy.asarray(self.target_matrix, dtype=theano.config.floatX), borrow=True)
        #indices_dev_theano=theano.shared(numpy.asarray(indices_dev, dtype=theano.config.floatX), borrow=True)
        indices_target_theano=T.cast(indices_target_theano, 'int32')
        #print 'context_matrix shape'
        #print self.context_matrix.shape[0], self.context_matrix.shape[1]
        #print self.context_matrix[:,0:300], self.context_matrix[:,300:600], self.context_matrix[:,600:900], self.context_matrix[:,900:]
        indices_context_theano=theano.shared(numpy.asarray(self.context_matrix, dtype=theano.config.floatX), borrow=True)
        #indices_dev_theano=theano.shared(numpy.asarray(indices_dev, dtype=theano.config.floatX), borrow=True)
        indices_context_theano=T.cast(indices_context_theano, 'int32')        

        #indices_dev_theano=T.cast(indices_dev_theano, 'int32')
        
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        x_index = T.imatrix('x_index')   # now, x is the index matrix, must be integer
        #y = T.ivector('y')  
        z = T.ivector('z')   # sentence length
        left=T.ivector('left')
        right=T.ivector('right')
        iteration= T.lscalar()
        t_index=T.imatrix('t_index')
        c_index=T.imatrix('c_index')
    
        x_index=debug_print(x_index,'x_index')
        x_transpose=debug_print(self.embeddings_R[x_index.flatten()].reshape((self.batch_size,self.maxSentLength, self.context_embedding_size)).transpose(0, 2, 1),'x_transpose')
        x=debug_print(x_transpose.flatten(),'x')
        ishape = (self.context_embedding_size, self.maxSentLength)  # this is the size of MNIST images
        filter_size1=(self.context_embedding_size,self.filter_size[0])
        filter_size2=(self.context_embedding_size/2,self.filter_size[1])
        #poolsize1=(1, ishape[1]-filter_size1[1]+1) #?????????????????????????????
        poolsize1=(1, ishape[1]+filter_size1[1]-1)
    
        '''
        left_after_conv=T.maximum(0,left-filter_size1[1]+1)
        right_after_conv=T.maximum(0, right-filter_size1[1]+1)
        '''
        left_after_conv=left
        right_after_conv=right
        
        #kmax=30 # this can not be too small, like 20
        #ktop=6
        #poolsize2=(1, kmax-filter_size2[1]+1) #(1,6)
        poolsize2=(1, self.kmax+filter_size2[1]-1) #(1,6)
        dynamic_lengths=T.maximum(self.ktop,z/2+1)  # dynamic k-max pooling
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
    
        # Reshape matrix of rasterized images of shape (batch_size,28*28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        layer0_input=debug_print(x.reshape((self.batch_size, 1, ishape[0], ishape[1])),'layer0_input')

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        '''
        layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                image_shape=(batch_size, 1, ishape[0], ishape[1]),
                filter_shape=(nkerns[0], 1, filter_size1[0], filter_size1[1]), poolsize=poolsize1, k=kmax)
        '''
        layer0 = Conv_Fold_DynamicK_PoolLayer(rng, input=layer0_input,
                image_shape=(self.batch_size, 1, ishape[0], ishape[1]),
                filter_shape=(self.nkerns[0], 1, filter_size1[0], filter_size1[1]), poolsize=poolsize1, k=dynamic_lengths, unifiedWidth=self.kmax, left=left_after_conv, right=right_after_conv, firstLayer=True)
        
        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
        '''
        layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                image_shape=(batch_size, nkerns[0], ishape[0], kmax),
                filter_shape=(nkerns[1], nkerns[0], filter_size2[0], filter_size2[1]), poolsize=poolsize2, k=ktop)
        '''
        '''
        left_after_conv=T.maximum(0, layer0.leftPad-filter_size2[1]+1)
        right_after_conv=T.maximum(0, layer0.rightPad-filter_size2[1]+1)
        '''
        left_after_conv=layer0.leftPad
        right_after_conv=layer0.rightPad
        dynamic_lengths=T.repeat([self.ktop],self.batch_size)  # dynamic k-max pooling
        layer1_input=debug_print(layer0.output, 'layer0_output')
        '''
        layer1 = ConvFoldPoolLayer(rng, input=layer0.output,
                image_shape=(batch_size, nkerns[0], ishape[0]/2, kmax),
                filter_shape=(nkerns[1], nkerns[0], filter_size2[0], filter_size2[1]), poolsize=poolsize2, k=ktop, left=left_after_conv, right=right_after_conv)
        '''
        layer1 = Conv_Fold_DynamicK_PoolLayer(rng, input=layer1_input,
                image_shape=(self.batch_size, self.nkerns[0], ishape[0]/2, self.kmax),
                filter_shape=(self.nkerns[1], self.nkerns[0], filter_size2[0], filter_size2[1]), poolsize=poolsize2, k=dynamic_lengths, unifiedWidth=self.ktop, left=left_after_conv, right=right_after_conv, firstLayer=False)    
        
        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        
        
        layer1_output = debug_print(layer1.output.flatten(2), 'layer1_output')
        #layer2_input=theano.printing.Print('layer2_input')(layer2_input)
        #produce sentence embeddings
        #layer2 = HiddenLayer(rng, input=layer2_input, n_in=self.nkerns[1] * (self.context_embedding_size/4) * self.ktop, n_out=self.sentEm_length, activation=T.tanh)
        
        #context_matrix,  target_matrix=self.extract_contexts_targets(indices_matrix=x_index, sentLengths=z, leftPad=left)
        target_matrix=t_index
        context_matrix=c_index
        #note that context indices might be zero embeddings
        h_indices=debug_print(context_matrix[:, self.context_size*iteration:self.context_size*(iteration+1)],'h_indices')
        w_indices=debug_print(target_matrix[:, iteration:(iteration+1)],'w_indices')
        #r_h is the concatenation of context embeddings
        r_h=debug_print(self.embed_context(h_indices), 'embedded_context')  #(batch_size, context_size*embedding_size)
        q_w=debug_print(self.embed_target(w_indices), 'embedded_target')
        #q_hat: concatenate sentence embeddings and context embeddings
        #q_hat=self.concatenate_sent_context(layer2.output, r_h)
        q_hat=self.concatenate_sent_context(layer1_output, r_h)
        layer3 = HiddenLayer(rng, input=q_hat, n_in=self.nkerns[1] * (self.context_embedding_size/4) * self.ktop+self.context_size*self.context_embedding_size, n_out=self.target_embedding_size, activation=T.tanh)
        layer3_output=debug_print(layer3.output, 'layer3.output')
        noise_indices, p_n_noise=self.get_noise()
        noise_indices=debug_print(noise_indices, 'noise_indices')
        #noise_indices=theano.printing.Print('noise_indices')(noise_indices)
        s_theta_data=debug_print(T.sum(layer3_output * q_w, axis=1).reshape((self.batch_size,1)) + self.bias[w_indices] , 's_theta_data')
        #s_theta_data=theano.printing.Print('s_theta_data')(s_theta_data)
        p_n_data = debug_print(self.p_n[w_indices],'p_n_data') #p_n[0] indicates the probability of word indexed 1
        delta_s_theta_data = debug_print(s_theta_data - T.log(self.k * p_n_data),'delta_s_theta_data')
        log_sigm_data = debug_print(T.log(T.nnet.sigmoid(delta_s_theta_data)),'log_sigm_data')
        
        #create the noise, q_noise has shape(self.batch_size, self.k, self.embedding_size )
        q_noise = debug_print(self.embed_noise(noise_indices),'embed_noise')
        q_hat_res = layer3_output.reshape((self.batch_size, 1, self.target_embedding_size))
        s_theta_noise = debug_print(T.sum(q_hat_res * q_noise, axis=2) + self.bias[noise_indices],'s_theta_noise') #(batch_size, k)
        delta_s_theta_noise = debug_print(s_theta_noise - T.log(self.k * p_n_noise), 'delta_s_theta_noise')  # it should be matrix (batch_size, k)
        log_sigm_noise = debug_print(T.log(1 - T.nnet.sigmoid(delta_s_theta_noise)), 'log_sigm_noise')
        sum_noise_per_example =debug_print(T.sum(log_sigm_noise, axis=1), 'sum_noise_per_example')   #(batch_size, 1)
        # Calc objective function
        J = debug_print(-T.mean(log_sigm_data) - T.mean(sum_noise_per_example),'J')
        L2_reg = (layer3.W** 2).sum()+ (layer1.W** 2).sum()+(layer0.W** 2).sum()+(self.embeddings_R**2).sum()#+( self.embeddings_Q**2).sum()
        self.cost = J + self.L2_weight*L2_reg
        '''
        validate_model = theano.function([index,iteration], self.cost,
                givens={
                    x_index: indices_dev_theano[index: index + self.batch_size],
                    z: devLengths[index: index + self.batch_size],
                    left: devLeftPad[index: index + self.batch_size],
                    right: devRightPad[index: index + self.batch_size]})
        '''
        # create a list of all model parameters to be fit by gradient descent
        self.params = layer3.params+layer1.params + layer0.params+[self.embeddings_R]#, self.embeddings_Q]
        #params = layer3.params + layer2.params + layer0.params+[embeddings]
        
        accumulator=[]
        for para_i in self.params:
            eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
            accumulator.append(theano.shared(eps_p, borrow=True))
          
        # create a list of gradients for all model parameters
        grads = T.grad(self.cost, self.params)
        updates = []
        for param_i, grad_i, acc_i in zip(self.params, grads, accumulator):
            grad_i=debug_print(grad_i,'grad_i')
            acc = acc_i + T.sqr(grad_i)
            if param_i == self.embeddings_R:# or param_i == self.embeddings_Q:
                updates.append((param_i, T.set_subtensor((param_i - self.ini_learning_rate * grad_i / T.sqrt(acc))[0], theano.shared(numpy.zeros(self.context_embedding_size)))))   #AdaGrad
            else:
                updates.append((param_i, param_i - self.ini_learning_rate * grad_i / T.sqrt(acc)))   #AdaGrad
            updates.append((acc_i, acc))    
           
        train_model = theano.function([index,iteration], [self.cost], updates=updates,
              givens={
                x_index: indices_train_theano[index: index + self.batch_size],
                z: trainLengths[index: index + self.batch_size],
                left: trainLeftPad[index: index + self.batch_size],
                right: trainRightPad[index: index + self.batch_size],
                t_index: indices_target_theano[index: index + self.batch_size],
                c_index: indices_context_theano[index: index + self.batch_size]})
    
        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        # early-stopping parameters
        patience = 50000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(10, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
        best_params = None
        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()
    
        epoch = 0
        done_looping = False
        vali_loss_list=[]
        train_loss_list=[]
        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            #for minibatch_index in xrange(n_train_batches): # each batch
            minibatch_index=0
            for batch_start in train_batch_start: 
                # iter means how many batches have been runed, taking into loop
                iter = (epoch - 1) * n_train_batches + minibatch_index +1
    
                minibatch_index=minibatch_index+1
                #print 'batch_start: '+str(batch_start)
                total_iteration=min(max(self.target_lengths[batch_start: batch_start + self.batch_size]), 60) # total iteration is not allowed to surpass 60
                # we only care the last cost within those iterations
                cost_of_end_batch=0.0
                costs_in_batch=[]
                for iteration in range(total_iteration):
                    #print 'iteration: '+str(iteration)+'/'+str(total_iteration)+' in iter '+str(iter)
                    #if iteration==3:
                    #    exit(0)
                    cost_of_end_batch = train_model(batch_start, iteration)
                    '''
                    print 'updated self.embeddings_R:'
                    print self.embeddings_R.get_value()[:37,:]
                    print self.embeddings_R.get_value()[37:,:]

                    print 'updated layer0 W: '
                    print layer0.W.get_value()[0:1,0:1,0:1,:]
                    print 'updated layer1 W:'
                    print layer1.W.get_value()[0:1,0:1,0:1,:]

                    print 'updated layer2 W: '
                    print layer2.W.get_value()
                    print 'updated layer3 W:'
                    print layer3.W.get_value()
                    '''
                    costs_in_batch.append(cost_of_end_batch)
                    #print 'cost_of_each_iteration: '+str(cost_of_end_batch)
                average_cost_per_batch=numpy.mean(costs_in_batch)
                #print 'cost_of_batch: '+str(average_cost_per_batch)
                if iter % validation_frequency == 0:
                    print 'training @ iter = '+str(iter)+' cost: '+str(average_cost_per_batch)# +' error: '+str(error_ij)
                    #print batch_embs
                #store sentence embeddings
                #for row in range(batch_start, batch_start + self.batch_size):
                #    sentences_embs[row]=batch_embs[row-batch_start]
                    
                if average_cost_per_batch<minimal_of_list(train_loss_list):
                    del train_loss_list[:]
                    train_loss_list.append(average_cost_per_batch)
                    self.best_params=self.params
                elif len(train_loss_list)<self.vali_cost_list_length:
                    train_loss_list.append(average_cost_per_batch)
                    if len(train_loss_list)==self.vali_cost_list_length:
                        self.store_model_to_file()
                        #self.store_sentence_embeddings(sentences_embs)
                        self.store_embeddings()
                        print 'Training over, best model got at train_cost:'+str(train_loss_list[0])
                        exit(0)                        
                #print 'sentence embeddings:'
                #print sentences_embs[:6,:]
                #if iter ==1:
                #    exit(0)
                '''
                if iter % validation_frequency == 0:
                    print 'training @ iter = '+str(iter)+' cost: '+str(cost_of_end_batch)# +' error: '+str(error_ij)
                if iter % validation_frequency == 0:
                    #print '\t iter: '+str(iter)
                    # compute zero-one loss on validation set
                    #validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                    validation_losses=[]
                    for batch_start in dev_batch_start:
                        #print '\t\t batch_start: '+str(batch_start)
                        total_iteration=max(self.dev_lengths[batch_start: batch_start + self.batch_size])
                        #for validate, we need the cost among all the iterations in that batch

                        for iteration in range(total_iteration):
                            vali_loss_i=validate_model(batch_start, iteration)
                            #print vali_loss_i
                            validation_losses.append(vali_loss_i)
                    this_validation_loss = numpy.mean(validation_losses)
                    print('\t\tepoch %i, minibatch %i/%i, validation cost %f ' % \
                      (epoch, minibatch_index , n_train_batches, \
                       this_validation_loss))
                    
                    if this_validation_loss < minimal_of_list(vali_loss_list):
                        del vali_loss_list[:]
                        vali_loss_list.append(this_validation_loss)
                        #store params
                        self.best_params=self.params
                        #fake
                    elif len(vali_loss_list)<self.vali_cost_list_length:
                        vali_loss_list.append(this_validation_loss)
                        if len(vali_loss_list)==self.vali_cost_list_length:
                            self.store_model_to_file()
                            self.store_sentence_embeddings(sentences_embs)
                            print 'Training over, best model got at vali_cost:'+str(vali_loss_list[0])
                            exit(0)
                '''
    
                if patience <= iter:
                    done_looping = True
                    break
    
        end_time = time.clock()
        '''
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i,'\
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        '''
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

    def store_sentence_embeddings(self, sentences_embs):
        if self.train_scheme ==1:
            save_file = open('/mounts/data/proj/wenpeng/CNN_LM/sentence_embeddings_HS.txt', 'w')  # this will overwrite current contents
        elif self.train_scheme ==2 :
            save_file = open('/mounts/data/proj/wenpeng/CNN_LM/sentence_embeddings_HS.txt', 'w')  # this will overwrite current contents
        for row in range(sentences_embs.shape[0]):
            for col in range(sentences_embs.shape[1]):
                save_file.write(str(sentences_embs[row, col])+" ")
            save_file.write("\n")
        save_file.close() 
        print 'Sentence embeddings stored over.'

def minimal_of_list(list_of_ele):
    if len(list_of_ele) ==0:
        return 1e10
    else:
        return list_of_ele[0]

   
    

if __name__ == '__main__':
    
    network=CNN_LM(learning_rate=0.001, n_epochs=2000, nkerns=[6, 14], batch_size=50, useAllSamples=0, ktop=4, filter_size=[7,5],
                    L2_weight=0.00005, dropout_p=0.8, useEmb=0, task=2, corpus=0, dataMode=2, maxSentLength=250, sentEm_length=200, window=10, 
                    k=10, nce_seeds=2345, only_left_context=False, vali_cost_list_length=20, context_embedding_size=50, train_scheme=1, max_size=1000000)
    network.evaluate_lenet5()

