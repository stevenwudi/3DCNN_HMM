# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from data import *
import numpy.random as nr
import numpy as n
import random as r
from util import *


class CIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = 32
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labels']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
class CroppedCIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.border_size = dp_params['crop_border']
        self.inner_size = 32 - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3
        
        for d in self.data_dic:
            d['data'] = n.require(d['data'], requirements='C')
            d['labels'] = n.require(n.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')
        
        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]

        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((3,32,32))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        cropped = self.cropped_data[self.batches_generated % 2]

        self.__trim_borders(datadic['data'], cropped)
        cropped -= self.data_mean
        self.batches_generated += 1
        return epoch, batchnum, [cropped, datadic['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
    def __trim_borders(self, x, target):
        y = x.reshape(3, 32, 32, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views/2):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                    target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))
    

class Kaggle_Galaxy_ConvNetDataProvider(LabeledDataProvider_Kaggle_Galaxy):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider_Kaggle_Galaxy.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = 224
        #reshape in python must be a tuple....
        self.data_mean = self.data_mean.reshape((self.data_mean.shape[0],1))
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        #for d in self.data_dic:
        #    # This converts the data matrix to single precision and makes sure that it is C-ordered
        #    d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
        #    d['labels'] = n.require(d['targets_all'].T, dtype=n.single, requirements='C')
        
    def get_next_batch(self, training = False):
        epoch, batchnum, datadic = LabeledDataProvider_Kaggle_Galaxy.get_next_batch(self)
        datadic['data'] = n.require((datadic['data'] - self.data_mean), dtype=n.single, requirements='C')
        datadic['labels'] = n.require(datadic['targets_all'].T, dtype=n.single, requirements='C')
        #return epoch, batchnum, [datadic['data'], datadic['labels']]
        # here we manually enhance the dataset by rotating and mirroring

        # sanity check, disappointing
        if training:
            enhance_number = 3
        else:
            enhance_number = 5
        im_rotated_array= n.empty((enhance_number*datadic['data'].shape[1], datadic['data'].shape[0]),dtype=n.single)
        labels = n.empty((enhance_number*datadic['data'].shape[1], datadic['labels'].shape[0]),dtype=n.single)
        for k in range(datadic['data'].shape[1]):
            im_rotated_array[ k*enhance_number: (k+1)*enhance_number, :] = image_rotate(datadic['data'][:,k], enhance_number, training)
            labels[ k*enhance_number: (k+1)*enhance_number, :] = n.tile(datadic['labels'][:,k],(enhance_number, 1))

        im_rotated_array = n.require(im_rotated_array.T, dtype=n.single, requirements='C')
        labels= n.require(labels.T, dtype=n.single, requirements='C')
        return epoch, batchnum, [im_rotated_array, labels], enhance_number
        
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['data_dim'] if idx == 0 else 37

      # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)


class Kaggle_Galaxy_Memory_ConvNetDataProvider(LabeledMemoryDataProvider_Kaggle_Galaxy):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider_Kaggle_Galaxy.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = 224
        #reshape in python must be a tuple....
        self.data_mean = self.data_mean.reshape((self.data_mean.shape[0],1))
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['targets_all'].T, dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider_Kaggle_Galaxy.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labels']]    
        
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['data_dim'] if idx == 0 else 37

      # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)



class MNIST_ConvNetDataProvider(LabeledMemoryDataProvider_MNIST):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider_MNIST.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 1
        self.img_size = 28
        #reshape in python must be a tuple....
        self.data_mean = self.data_mean.reshape((self.data_mean.shape[0],1))
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider_MNIST.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labels']] 
        
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['data_dim'] if idx == 0 else 1

      # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

class AVLETTERS_ConvNetDataProvider(LabeledMemoryDataProvider_AVLETTERS):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider_AVLETTERS.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 5
        self.img_size = 60
        #reshape in python must be a tuple....
        self.data_mean = self.data_mean.reshape((self.data_mean.shape[0],1))
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'].T - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')



    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider_AVLETTERS.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labels']] 
        
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['data_dim'] if idx == 0 else 1

      # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

class CodaLab_MemoryConvNetDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 4
        self.img_size = 90
        self.data_dic = []
        self.data_mean = self.data_mean.reshape((90*90*4,1))
        # uncommment following for the loading all the data into memory (which is bad for big data)
        for i in batch_range:
            self.data_dic += [unpickle(self.get_data_file_name(i))]
            self.data_dic[-1]['data'] = n.require((self.data_dic[-1]['data'] - self.data_mean), dtype=n.single, requirements='C')
            mask = (self.data_dic[-1]['data_id'] ==201)
            self.data_dic[-1]['data_id'][mask] = 200 #stupid mistakes made by Di Wu
            self.data_dic[-1]["labels"] = n.c_[n.require(self.data_dic[-1]['data_id'].T, dtype=n.single)]
        #reshape in python must be a tuple....



    def get_next_batch(self, training = False):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        bidx = batchnum - self.batch_range[0]
        return epoch, batchnum, [self.data_dic[bidx]['data'],self.data_dic[bidx]['labels']]     

    def get_data_dims(self, idx=0):
        return self.batch_meta['data_dim'] if idx == 0 else 1

      # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

    def get_num_classes(self):
        # we need to change to 201 when there is a neural pose
        return 201    

class CodaLab_ConvNetDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 4
        self.img_size = 90
        self.data_dic = []
        self.data_mean = self.data_mean.reshape((4*90*90,1))

    def get_next_batch(self, training = False):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        bidx = batchnum - self.batch_range[0]

        self.data_dic = unpickle(self.get_data_file_name(bidx+1))
        self.data_dic['data'] = n.require((self.data_dic['data'] - self.data_mean), dtype=n.single, requirements='C')
        
        #a=self.data_dic['data'][0:90*90,0]
        #im = a.reshape((90,90))
        #from matplotlib import pylab
        #pylab.imshow(im)
        #pylab.show()
        
        mask = (self.data_dic['data_id'] ==201)
        self.data_dic['data_id'][mask] = 200 #stupid mistakes made by Di Wu
        self.data_dic["labels"] = n.c_[n.require(self.data_dic['data_id'].T, dtype=n.single)]

        return epoch, batchnum, [self.data_dic['data'],self.data_dic['labels']]      


    def get_data_dims(self, idx=0):
        return self.batch_meta['data_dim'] if idx == 0 else 1


    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

    def get_num_classes(self):
        # we need to change to 201 when there is a neural pose
        return 201    




class CroppedCodaLab_ConvNetDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.border_size = dp_params['crop_border']
        self.img_size = 90
        self.inner_size = self.img_size - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 4
        self.data_mean = self.batch_meta['data_mean'].reshape((4,90,90))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self, training = False):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        bidx = batchnum - self.batch_range[0]

        self.data_dic = unpickle(self.get_data_file_name(bidx+1))

        mask = (self.data_dic['data_id'] ==201)
        self.data_dic['data_id'][mask] = 200 #stupid mistakes made by Di Wu
        self.data_dic["labels"] = n.c_[n.require(self.data_dic['data_id'].T, dtype=n.single)]
        self.data_dic['labels'] = n.require(n.tile(self.data_dic['labels'].reshape((1, self.data_dic['data'].shape[1])), (1, self.data_mult)), requirements='C')


        self.data_dic['data'] = n.require(self.data_dic['data'], dtype=n.single, requirements='C')
        self.cropped_data = n.zeros((self.get_data_dims(), self.data_dic['data'].shape[1]*self.data_mult), dtype=n.single)
        cropped = self.cropped_data
        self.__trim_borders(self.data_dic['data'], cropped)
        cropped -= self.data_mean

        return epoch, batchnum, [cropped, self.data_dic['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 4 if idx == 0 else 1

    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
    def __trim_borders(self, x, target):
        y = x.reshape(self.num_colors, self.img_size, self.img_size, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                #if nr.randint(2) == 0: # also flip the image with 50% probability
                #    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))

    def get_num_classes(self):
        # we need to change to 201 when there is a neural pose
        return 201   