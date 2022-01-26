import time
import pandas as pd
from joblib import Parallel,delayed
import numpy as np
import cv2
import queue
import glob
import os
import time
import scipy.ndimage
import joblib
import sys
import matplotlib.pyplot as plt
import niqe 
import save_stats
from numba import jit,prange,njit
import argparse

os.nice(1)

parser = argparse.ArgumentParser(description='Generate ChipQA features from a folder of videos and store them')
parser.add_argument('--input_folder',help='Folder containing input videos')
parser.add_argument('--results_folder',help='Folder where features are stored')

args = parser.parse_args()
C=1


def fread(fid, nelements, dtype):
     if dtype is np.str:
         dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
     else:
         dt = dtype

     data_array = np.fromfile(fid, dt, nelements)
     data_array.shape = (nelements, 1)

     return data_array



def yuv_read(file_object,frame_num,height,width):
    file_object.seek(frame_num*height*width*1.5)
    y1 = fread(file_object,height*width,np.uint8)
    u1 = fread(file_object,height*width//4,np.uint8)
    v1 = fread(file_object,height*width//4,np.uint8)
    y = np.reshape(y1,(height,width))
    u = np.reshape(u1,(height//2,width//2)).repeat(2,axis=0).repeat(2,axis=1)
    v = np.reshape(v1,(height//2,width//2)).repeat(2,axis=0).repeat(2,axis=1)
    return y.astype(np.float32),u.astype(np.float32),v.astype(np.float32)

def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights
def compute_image_mscn_transform(image, C=1e-3, avg_window=None, extend_mode='constant'):
    if avg_window is None:
      avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    return (image - mu_image)/(var_image + C), var_image, mu_image

def spatiotemporal_mscn(img_buffer,avg_window,extend_mode='mirror'):
    st_mean = np.zeros((img_buffer.shape))
    scipy.ndimage.correlate1d(img_buffer, avg_window, 0, st_mean, mode=extend_mode)
    return st_mean

@jit(nopython=True)
def find_sts_locs(sts_slope,cy,cx,step,h,w):
    if(np.abs(sts_slope)<1):
        x_sts = np.arange(cx-int((step-1)/2),cx+int((step-1)/2)+1)
        y = (cy-(x_sts-cx)*sts_slope).astype(np.int64)
        y_sts = np.asarray([y[j] if y[j]<h else h-1 for j in range(step)])
    else:
        #        print(np.abs(sts_slope))
        y_sts = np.arange(cy-int((step-1)/2),cy+int((step-1)/2)+1)
        x= ((-y_sts+cy)/sts_slope+cx).astype(np.int64)
        x_sts = np.asarray([x[j] if x[j]<w else w-1 for j in range(step)]) 
    return x_sts,y_sts


@jit(nopython=True)
def find_kurtosis_slice(Y3d_mscn,cy,cx,rst,rct,theta,h,step):
    st_kurtosis = np.zeros((len(theta),))
    data = np.zeros((len(theta),step**2))
    for index,t in enumerate(theta):
        rsin_theta = rst[:,index]
        rcos_theta  =rct[:,index]
        x_sts,y_sts = cx+rcos_theta,cy+rsin_theta
        
        data[index,:] =Y3d_mscn[:,y_sts*h+x_sts].flatten() 
        data_mu4 = np.mean((data[index,:]-np.mean(data[index,:]))**4)
        data_var = np.var(data[index,:])
        st_kurtosis[index] = data_mu4/(data_var**2+1e-4)
    idx = (np.abs(st_kurtosis - 3)).argmin()
    
    data_slice = data[idx,:]
    return data_slice,st_kurtosis[idx]-3


def find_kurtosis_sts(img_buffer,grad_img_buffer,step,cy,cx,rst,rct,theta):

    h, w = img_buffer[step-1].shape[:2]
    Y3d_mscn = np.reshape(img_buffer.copy(),(step,-1))
    gradY3d_mscn = np.reshape(grad_img_buffer.copy(),(step,-1))
    sts= [find_kurtosis_slice(Y3d_mscn,cy[i],cx[i],rst,rct,theta,h,step) for i in range(len(cy))]
    sts_grad= [find_kurtosis_slice(gradY3d_mscn,cy[i],cx[i],rst,rct,theta,h,step) for i in range(len(cy))]

    st_data = [sts[i][0] for i in range(len(sts))]
    st_deviation = [sts[i][1] for i in range(len(sts))]
    st_grad_data = [sts_grad[i][0] for i in range(len(sts_grad))]
    st_grad_dev = [sts_grad[i][1] for i in range(len(sts_grad))]
    return st_data,np.asarray(st_deviation),st_grad_data,np.asarray(st_grad_dev)



def sts_fromfilename(i,filenames,results_folder):
    filename = filenames[i]
    print(filename)
    if(os.path.exists(filename)==False):
        return
    name = os.path.basename(filename)
    print(name) 
    filename_out =os.path.join(results_folder,os.path.splitext(name)[0]+'.z')
    if(os.path.exists(filename_out)):
        return
    ## PARAMETERS for the model
    st_time_length = 5
    t = np.arange(0,st_time_length)
    a=0.5
    # temporal filter
    avg_window = t*(1-a*t)*np.exp(-2*a*t)
    avg_window = np.flip(avg_window)

    # LUT for coordinate search
    theta = np.arange(0,np.pi,np.pi/6)
    ct = np.cos(theta)
    st = np.sin(theta)
    lower_r = int((st_time_length+1)/2)-1
    higher_r = int((st_time_length+1)/2)
    r = np.arange(-lower_r,higher_r)
    rct = np.round(np.outer(r,ct))
    rst = np.round(np.outer(r,st))
    rct = rct.astype(np.int32)
    rst = rst.astype(np.int32)

    h,w = 2160,3840
    #percent by which the image is resized
    scale_percent = 0.5
    # dsize
    dsize = (int(scale_percent*h),int(scale_percent*w))

    # opening file object
    dis_file_object = open(filename)
    dis_file_object.seek(0, os.SEEK_END)
    dist_filesize = dis_file_object.tell()
    multiplier = 1.5
    framenos= int(dist_filesize/(h*w*multiplier))

    prevY,_,_ = yuv_read(dis_file_object,0,h,w)
    prevY = prevY.astype(np.float32)


    # ST chip centers and parameters
    step = st_time_length
    cy, cx = np.mgrid[step:h-step*4:step*4, step:w-step*4:step*4].reshape(2,-1).astype(int) # these will be the centers of each block
    dcy, dcx = np.mgrid[step:dsize[0]-step*4:step*4, step:dsize[1]-step*4:step*4].reshape(2,-1).astype(int) # these will be the centers of each block
    r1 = len(np.arange(step,h-step*4,step*4)) 
    r2 = len(np.arange(step,w-step*4,step*4)) 
    dr1 = len(np.arange(step,dsize[0]-step*4,step*4)) 
    dr2 = len(np.arange(step,dsize[1]-step*4,step*4)) 

    
    C = 1
    prevY_down = cv2.resize(prevY,(dsize[1],dsize[0]),interpolation=cv2.INTER_CUBIC)

    print(prevY.shape,prevY_down.shape)
    img_buffer = np.zeros((st_time_length,prevY.shape[0],prevY.shape[1]))
    grad_img_buffer = np.zeros((st_time_length,prevY.shape[0],prevY.shape[1]))
    down_img_buffer =np.zeros((st_time_length,prevY_down.shape[0],prevY_down.shape[1]))
    graddown_img_buffer =np.zeros((st_time_length,prevY_down.shape[0],prevY_down.shape[1]))


    gradient_x = cv2.Sobel(prevY,ddepth=-1,dx=1,dy=0)
    gradient_y = cv2.Sobel(prevY,ddepth=-1,dx=0,dy=1)
    gradient_mag = np.sqrt(gradient_x**2+gradient_y**2)    

    
    gradient_x_down = cv2.Sobel(prevY_down,ddepth=-1,dx=1,dy=0)
    gradient_y_down = cv2.Sobel(prevY_down,ddepth=-1,dx=0,dy=1)
    gradient_mag_down = np.sqrt(gradient_x_down**2+gradient_y_down**2)    
    i = 0

    Y_mscn,_,_ = compute_image_mscn_transform(prevY,C)
    dY_mscn,_,_ = compute_image_mscn_transform(prevY_down,C)
    gradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag,C)
    dgradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag_down,C)

    img_buffer[i,:,:] = Y_mscn
    down_img_buffer[i,:,:]= dY_mscn
    grad_img_buffer[i,:,:] =gradY_mscn 
    graddown_img_buffer[i,:,:]=dgradY_mscn 
    i = i+1

    
    head, tail = os.path.split(filename)


    
    spat_list = []
    X_list = []
    spatavg_list = []
    feat_sd_list =  []
    sd_list= []
    
    j=0
    total_time = 0
    for framenum in range(1,framenos): 
#        print(framenum)
        # uncomment for FLOPS
        #high.start_counters([events.PAPI_FP_OPS,])
        
        try:
            Y,U,V = yuv_read(dis_file_object,framenum,h,w)
            
        except:
            f = open("chipqa_yuv_reading_error.txt", "a")
            f.write(filename+"\n")
            f.close()
            break
        YUV =np.stack((Y,U,V),2)
        
        BGR =  cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR)
        lab = cv2.cvtColor(BGR, cv2.COLOR_BGR2LAB)
        lab = lab.astype(np.float32)
        chroma_feats = save_stats.chroma_feats(lab)

        Y = Y.astype(np.float32)
        Y_down = cv2.resize(Y,(dsize[1],dsize[0]),interpolation=cv2.INTER_CUBIC)
        
        
        gradient_x = cv2.Sobel(Y,ddepth=-1,dx=1,dy=0)
        gradient_y = cv2.Sobel(Y,ddepth=-1,dx=0,dy=1)
        gradient_mag = np.sqrt(gradient_x**2+gradient_y**2)    

        
        gradient_x_down = cv2.Sobel(Y_down,ddepth=-1,dx=1,dy=0)
        gradient_y_down = cv2.Sobel(Y_down,ddepth=-1,dx=0,dy=1)
        gradient_mag_down = np.sqrt(gradient_x_down**2+gradient_y_down**2)    


        Y_mscn,Ysigma,_ = compute_image_mscn_transform(Y,C)
        dY_mscn,dYsigma,_ = compute_image_mscn_transform(Y_down,C)

        gradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag,C)
        dgradY_mscn,_,_ = compute_image_mscn_transform(gradient_mag_down,C)

        gradient_feats = save_stats.extract_secondord_feats(gradY_mscn)
        gdown_feats = save_stats.extract_secondord_feats(dgradY_mscn)
        gfeats = np.concatenate((gradient_feats,gdown_feats),axis=0)

        
        Ysigma_mscn,_,_= compute_image_mscn_transform(Ysigma,C)
        dYsigma_mscn,_,_= compute_image_mscn_transform(dYsigma,C)

        sigma_feats = save_stats.stat_feats(Ysigma_mscn)
        dsigma_feats = save_stats.stat_feats(dYsigma_mscn)


        feats = np.concatenate((chroma_feats,gfeats,sigma_feats,dsigma_feats),axis=0)

        feat_sd_list.append(feats)
        spatavg_list.append(feats)

        
        img_buffer[i,:,:] = Y_mscn
        down_img_buffer[i,:,:]= dY_mscn
        grad_img_buffer[i,:,:] =gradY_mscn 
        graddown_img_buffer[i,:,:]=dgradY_mscn 
        i=i+1


        if (i>=st_time_length): 

            Y3d_mscn = spatiotemporal_mscn(img_buffer,avg_window)
            Ydown_3d_mscn = spatiotemporal_mscn(down_img_buffer,avg_window)
            grad3d_mscn = spatiotemporal_mscn(grad_img_buffer,avg_window)
            graddown3d_mscn = spatiotemporal_mscn(graddown_img_buffer,avg_window)
            spat_feats = niqe.compute_niqe_features(Y,C=C)

            sd_feats = np.std(feat_sd_list,axis=0)
            sd_list.append(sd_feats)
            feat_sd_list = []

            sts,st_deviation,sts_grad,sts_grad_deviation = find_kurtosis_sts(Y3d_mscn,grad3d_mscn,step,cy,cx,rst,rct,theta)
            dsts,dsts_deviation,dsts_grad,dsts_grad_deviation = find_kurtosis_sts(Ydown_3d_mscn,graddown3d_mscn,step,dcy,dcx,rst,rct,theta)
            sts_arr = np.reshape(sts,(r1*st_time_length,r2*st_time_length)) 
            sts_grad = np.reshape(sts_grad,(r1*st_time_length,r2*st_time_length))
            dsts_arr = np.reshape(dsts,(dr1*st_time_length,dr2*st_time_length)) #(int((int(dsize[0]/20)*20-step*4)/4),int((int(dsize[1]/20)*20-step*4)/4)))
            dsts_grad = np.reshape(dsts_grad,(dr1*st_time_length,dr2*st_time_length))#(int((int(dsize[0]/20)*20-step*4)/4),int((int(dsize[1]/20)*20-step*4)/4)))
            feats =  save_stats.brisque(sts_arr)
            grad_feats = save_stats.brisque(sts_grad)
            
            dfeats =  save_stats.brisque(dsts_arr)
            dgrad_feats = save_stats.brisque(dsts_grad)


            allst_feats = np.concatenate((spat_feats,feats,dfeats,grad_feats,dgrad_feats),axis=0)
            X_list.append(allst_feats)


            img_buffer = np.zeros((st_time_length,prevY.shape[0],prevY.shape[1]))
            grad_img_buffer = np.zeros((st_time_length,prevY.shape[0],prevY.shape[1]))
            down_img_buffer =np.zeros((st_time_length,prevY_down.shape[0],prevY_down.shape[1]))
            graddown_img_buffer =np.zeros((st_time_length,prevY_down.shape[0],prevY_down.shape[1]))
            i=0
#            x=high.stop_counters()
#        print(x,"is the number of flops")

    X1 = np.average(spatavg_list,axis=0)
    X2 = np.average(sd_list,axis=0)
    X3 = np.average(X_list,axis=0)
    X = np.concatenate((X1,X2,X3),axis=0)
    train_dict = {"features":X}
    filename_out =os.path.join(results_folder,os.path.splitext(name)[0]+'.z')
    joblib.dump(train_dict,filename_out)
    return


def sts_fromvid(args):
    files = glob.glob(os.path.join(args.input_folder,'*.yuv'))
    
    outfolder = args.results_folder
    if(os.path.exists(outfolder)==False):
        os.mkdir(outfolder)
    Parallel(n_jobs=5)(delayed(sts_fromfilename)\
            (i,files,outfolder)\
            for i in range(len(files)))
#    for i in range(len(files)):
#        sts_fromfilename(i,files,framenos_list,args.results_folder,ws,hs,nl_method='nakarushton',use_csf=False,use_lnl=False)
             



    return


def main():
    args = parser.parse_args()
    sts_fromvid(args)


if __name__ == '__main__':
    # print(__doc__)
    main()
    
