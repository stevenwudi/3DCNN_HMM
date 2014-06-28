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

import re
import cPickle
import os
import numpy as n
from math import sqrt
import random

import gzip
import zipfile

class UnpickleError(Exception):
    pass

VENDOR_ID_REGEX = re.compile('^vendor_id\s+: (\S+)')
GPU_LOCK_NO_SCRIPT = -2
GPU_LOCK_NO_LOCK = -1

try:
    import magic
    ms = magic.open(magic.MAGIC_NONE)
    ms.load()
except ImportError: # no magic module
    ms = None

def get_gpu_lock(id=-1):
    import imp
    lock_script_path = '/u/tang/bin/gpu_lock2.py'
    if os.path.exists(lock_script_path):
        locker = imp.load_source("", lock_script_path)
        if id == -1:
            return locker.obtain_lock_id()
        print id
        got_id = locker._obtain_lock(id)
        return id if got_id else GPU_LOCK_NO_LOCK
    return GPU_LOCK_NO_SCRIPT if id < 0 else id

def pickle(filename, data, compress=False):
    if compress:
        fo = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        fo.writestr('data', cPickle.dumps(data, -1))
    else:
        fo = open(filename, "wb")
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()
    
def unpickle(filename):
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)
    if ms is not None and ms.file(filename).startswith('gzip'):
        fo = gzip.open(filename, 'rb')
        dict = cPickle.load(fo)
    elif ms is not None and ms.file(filename).startswith('Zip'):
        fo = zipfile.ZipFile(filename, 'r', zipfile.ZIP_DEFLATED)
        dict = cPickle.loads(fo.read('data'))
    else:
        fo = open(filename, 'rb')
        dict = cPickle.load(fo)
    
    fo.close()
    return dict

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def is_intel_machine():
    f = open('/proc/cpuinfo')
    for line in f:
        m = VENDOR_ID_REGEX.match(line)
        if m:
            f.close()
            return m.group(1) == 'GenuineIntel'
    f.close()
    return False

def get_cpu():
    if is_intel_machine():
        return 'intel'
    return 'amd'

def is_windows_machine():
    return os.name == 'nt'



def normalize_probs_Galaxy(targets):
    # Tree structure:
    tree = [
        [0, 1, 2], # 1.1 - 1.3,
        [3, 4], # 2.1 - 2.2
        [5, 6], # 3.1 - 3.2
        [7, 8], # 4.1 - 4.2
        [9, 10, 11, 12], # 5.1 -5.4
        [13, 14], # 6.1 - 6.2
        [15, 16, 17], # 7.1- 7.3
        [18, 19, 20, 21, 22, 23, 24], # 8.1 - 8.7
        [25, 26, 27], # 9.1- 9.3
        [28, 29, 30], # 10.1- 10.3
        [31, 32, 33, 34, 35, 36], # 11.1- 11.6
        ]

    # Tree parent
    parent_tree= [
        0,       # Q1
        tree[0][1], #Q2
        tree[1][1], #Q3
        tree[1][1], #Q4
        tree[1][1], #Q5
        0,          #Q6
        tree[0][0], #Q7
        tree[5][0], #Q8
        tree[1][0], #Q9
        tree[3][0], #Q10
        tree[3][0], #Q11
        ]

    for k, sum_index in enumerate(tree):
        if k==0 or k==5:
            actual_sums = targets[:, tree[k]].sum(1)
            desired_sums = 1
            den = (desired_sums/ (actual_sums+0.00001))
            targets[:, tree[k]] = targets[:, tree[k]] *  den [:,n.newaxis]
        else:
            actual_sums = targets[:, tree[k]].sum(1)
            desired_sums = targets[:, parent_tree[k]]
            den = (desired_sums/ (actual_sums+0.00001))
            targets[:, tree[k]] = targets[:, tree[k]] *  den [:,n.newaxis]

    return targets

def image_rotate(im_array,enhance_number, training = False):
    """A wrapper for generating multiple rotational images """     
        #2. we rotate the image 3 times with one mirrored image,
        #so we have 5 times more expanded training images   
        
        # but unfortunately, GPU can only take toughly 3*1000 (224*224*3) images per batch
        # So we need to choose only 3 images  

    arr_centered = (im_array.reshape(3,224,224)).T
    arr_mirrored = n.fliplr(arr_centered)      
              
    if training: # if it's training, we will randomly generate a number from 1-3
        rotation_degree = random.randint(1,3)
        arr_rotate = n.rot90(arr_centered, rotation_degree)
        data = n.empty((enhance_number, im_array.shape[0]),dtype=n.float32)
        data[0,:] =arr_centered.T.flatten('C')
        data[1,:] = arr_mirrored.T.flatten('C')
        data[2,:] = arr_rotate.T.flatten('C') 
    else: # but if it's testing, temporarily, we want to have a consistently testing errror
        data = n.empty((enhance_number, im_array.shape[0]),dtype=n.float32)
        arr_rotate_1 = n.rot90(arr_centered, 1)
        arr_rotate_2 = n.rot90(arr_centered, 2)
        arr_rotate_3 = n.rot90(arr_centered, 3)
        data[0,:] =arr_centered.T.flatten('C')
        data[1,:] = arr_mirrored.T.flatten('C')
        data[2,:] = arr_rotate_1.T.flatten('C')  
        data[3,:] = arr_rotate_2.T.flatten('C')
        data[4,:] = arr_rotate_3.T.flatten('C')  

    return data[0:enhance_number,:]