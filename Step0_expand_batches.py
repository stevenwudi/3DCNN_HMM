import sys, os
import cPickle
import numpy as np



data=os.path.join(".\storage_sk_normalize\\")
samples=os.listdir(data)

save_dir = '.\storage_sk_final\\'

data_neutral = os.path.join(".\storage_neutral_sk_normalize\\")
samples_neutral = os.listdir(data_neutral)

data_mean_all = np.zeros((4*90*90, len(samples)))

for file_count, file_name in enumerate(samples):
    #if file_count>0 : #because the first sample is meta data
        # first load data frame
        print "Dealing with file: " + file_name + " neutral file: " + samples_neutral[file_count]
        load_file= os.path.join(data,file_name)
        fp = open(load_file, 'rb')
        dic = cPickle.load(fp)
        fp.close()
        # then load neutral frames
        load_file= os.path.join(data_neutral,samples_neutral[file_count])
        fp = open(load_file, 'rb')
        dic_neutral = cPickle.load(fp)
        fp.close()
        # we divide each batch into 5 mini batches

        # we first extract some random 500 neutral examples:
        ### NOTE FROM CUDA_CNVNET: he data matrix must be C-ordered 
        #(this is the default for numpy arrays), with dimensions 
        #(data dimensionality)x(number of cases). !!!!!!!!!!!!!!
        #If your images have channels (for example, the 3 color channels in the CIFAR-10),
       # then all the values for channel n must precede all the values for channel n + 1.
       ##################################
        random_idx = np.random.permutation(range(dic_neutral['data'].shape[-1]))
        #(data dimensionality)x(number of cases). !!!!!!!!!!!!!!
        neutral_pose_temp = np.transpose(dic_neutral['data'][:,:,:, random_idx], (2,0,1,3))
        neutral_pose = np.reshape(neutral_pose_temp, (4*90*90, len(random_idx)) )
        neutral_label = dic_neutral['data_id'][ random_idx]

        #im = neutral_pose[0:90*90,0].reshape((90,90))
        #from matplotlib import pylab
        #pylab.imshow(im)
        #pylab.show()


        random_idx = np.random.permutation(range(dic['data'].shape[-1]))
        #(data dimensionality)x(number of cases). !!!!!!!!!!!!!!
        actual_pose_temp = np.transpose(dic['data'][:,:,:, random_idx], (2,0,1,3))
        actual_pose = np.reshape(actual_pose_temp, (4*90*90, len(random_idx)) )
        actual_label = dic['data_id'][ random_idx]

        #im = actual_pose[0:90*90,0].reshape((90,90))
        #from matplotlib import pylab
        #pylab.imshow(im)
        #pylab.show()

        data_temp = np.concatenate((actual_pose, neutral_pose), axis = 1)
        data_id = np.concatenate((actual_label, neutral_label))

        random_idx = np.random.permutation(range(data_temp.shape[-1]))
        pose_all = data_temp[:,random_idx] 
        label_all = data_id[random_idx]

        batch_num = file_count + 1

        save_path= os.path.join(save_dir,'data_batch_'+str(batch_num))
        out_file = open(save_path, 'wb')
        dic_temp = {'batch_label':['batch 1 of'+ str(batch_num)], 'data':pose_all, 
                        'data_id':label_all}
        cPickle.dump(dic_temp, out_file, protocol=cPickle.HIGHEST_PROTOCOL)
        out_file.close()

        data_mean_all[:, file_count]  = np.mean(pose_all, axis=1)

data_mean = data_mean_all.mean(axis=1)
dic = {'data_dim': 4*90*90,'data_in_rows':True, 'data_mean': data_mean}
file_name ='.\storage_sk_final\\batches.meta'
out_file = open(file_name, 'wb')
cPickle.dump(dic, out_file)
out_file.close()