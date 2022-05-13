import time
from yuv_utils import yuv_read
import numpy as np
import cv2
import queue
import glob
import os
import time
import scipy.ndimage
import joblib
import sys
import niqe 
import save_stats
from numba import jit,prange
import argparse

parser = argparse.ArgumentParser(description='Generate ChipQA features from a video and store them')
parser.add_argument('--input_file',help='Input video file')
parser.add_argument('--results_file',help='File where features are stored')
parser.add_argument('--width', type=int)
parser.add_argument('--height', type=int)
parser.add_argument('--bit_depth', type=int,choices={8,10,12})
parser.add_argument('--color_space',choices={'BT2020','BT709'})

args = parser.parse_args()
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

def spatiotemporal_mscn(img_buffer,avg_window,extend_mode='mirror'):
    st_mean = np.zeros((img_buffer.shape))
    scipy.ndimage.correlate1d(img_buffer, avg_window, 0, st_mean, mode=extend_mode)
    return st_mean

@jit(nopython=True)
def find_sts_locs(sts_slope,cy,cx,step,height,width):
    if(np.abs(sts_slope)<1):
        x_sts = np.arange(cx-int((step-1)/2),cx+int((step-1)/2)+1)
        y = (cy-(x_sts-cx)*sts_slope).astype(np.int64)
        y_sts = np.asarray([y[j] if y[j]<height else height-1 for j in range(step)])
    else:
        #        print(np.abs(sts_slope))
        y_sts = np.arange(cy-int((step-1)/2),cy+int((step-1)/2)+1)
        x= ((-y_sts+cy)/sts_slope+cx).astype(np.int64)
        x_sts = np.asarray([x[j] if x[j]<width else width-1 for j in range(step)]) 
    return x_sts,y_sts


@jit(nopython=True)
def find_kurtosis_slice(Y3d_mscn,cy,cx,rst,rct,theta,height,step):
    st_kurtosis = np.zeros((len(theta),))
    data = np.zeros((len(theta),step**2))
    for index,t in enumerate(theta):
        rsin_theta = rst[:,index]
        rcos_theta  =rct[:,index]
        x_sts,y_sts = cx+rcos_theta,cy+rsin_theta
        
        data[index,:] =Y3d_mscn[:,y_sts*height+x_sts].flatten() 
        data_mu4 = np.mean((data[index,:]-np.mean(data[index,:]))**4)
        data_var = np.var(data[index,:])
        st_kurtosis[index] = data_mu4/(data_var**2+1e-4)
    idx = (np.abs(st_kurtosis - 3)).argmin()
    
    data_slice = data[idx,:]
    return data_slice


def find_kurtosis_sts(img_buffer,grad_img_buffer,step,cy,cx,rst,rct,theta):

    height, width = img_buffer[step-1].shape[:2]
    Y3d_mscn = np.reshape(img_buffer.copy(),(step,-1))
    gradY3d_mscn = np.reshape(grad_img_buffer.copy(),(step,-1))
    sts= [find_kurtosis_slice(Y3d_mscn,cy[i],cx[i],rst,rct,theta,width,step) for i in range(len(cy))]
    sts_grad= [find_kurtosis_slice(gradY3d_mscn,cy[i],cx[i],rst,rct,theta,width,step) for i in range(len(cy))]

    return sts,sts_grad


def unblockshaped(arr, height, width):
    """
    Return an array of shape (height, width) where
    height * width = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(height//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(height, width))


def sts_fromfilename(filename,filename_out,height,width,bit_depth,color_space):
    name = os.path.basename(filename)
    print(name) 
    st_time_length = 5
    t = np.arange(0,st_time_length)
    a=0.5
    avg_window = t*(1-a*t)*np.exp(-2*a*t)
    avg_window = np.flip(avg_window)
                    #percent by which the image is resized
    scale_percent = 0.5
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


    vid_stream = open(filename,'r')
    vid_stream.seek(0, os.SEEK_END)
    vid_filesize = vid_stream.tell()
    if(bit_depth==8):
        multiplier =1.5
        C = 1
        color_C = 1
    elif(bit_depth==10):
        multiplier = 3
        C = 4
        color_C = 0.001

    vid_T = int(vid_filesize/(height*width*multiplier))
    framenum = 0
    prevY,U_pq,V_pq = yuv_read(filename,framenum,height,width,bit_depth)

    # dsize
    dsize = (int(scale_percent*height),int(scale_percent*width))
    print(height,width,dsize)

    step = st_time_length
    cy, cx = np.mgrid[step:height-step*4:step*4, step:width-step*4:step*4].reshape(2,-1).astype(int) # these will be the centers of each block
    dcy, dcx = np.mgrid[step:dsize[0]-step*4:step*4, step:dsize[1]-step*4:step*4].reshape(2,-1).astype(int) # these will be the centers of each block

    
    prevY_down = cv2.resize(prevY,(dsize[1],dsize[0]),interpolation=cv2.INTER_CUBIC)

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

    Y_mscn,_,_ = save_stats.compute_image_mscn_transform(prevY,C)
    dY_mscn,_,_ = save_stats.compute_image_mscn_transform(prevY_down,C)
    gradY_mscn,_,_ = save_stats.compute_image_mscn_transform(gradient_mag,C)
    dgradY_mscn,_,_ = save_stats.compute_image_mscn_transform(gradient_mag_down,C)

    img_buffer[i,:,:] = Y_mscn
    down_img_buffer[i,:,:]= dY_mscn
    grad_img_buffer[i,:,:] =gradY_mscn 
    graddown_img_buffer[i,:,:]=dgradY_mscn 
    i = i+1

    r1 = len(np.arange(step,height-step*4,step*4)) 
    r2 = len(np.arange(step,width-step*4,step*4)) 
    dr1 = len(np.arange(step,dsize[0]-step*4,step*4)) 
    dr2 = len(np.arange(step,dsize[1]-step*4,step*4)) 
    
    X_list = []
    spatavg_list = []
    feat_sd_list =  []
    sd_list= []
    
    j=0
    for framenum in range(1,vid_T):
        
        # uncomment for FLOPS
        #high.start_counters([events.PAPI_FP_OPS,])

        
        Y,U,V = yuv_read(filename,framenum,height,width,bit_depth)
        
        
        if(color_space=='BT709'):
            yvu = np.dstack((Y,V,U))
            bgr = cv2.cvtColor(yvu,cv2.COLOR_YCrCb2BGR)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            lab = lab.astype(np.float32)
        elif(color_space=='BT2020'):

            yuv = np.dstack((Y,U,V))
            frame = colour.YCbCr_to_RGB(yuv/1023.0,K = [0.2627,0.0593])
            xyz = colour.RGB_to_XYZ(frame, [0.3127,0.3290], [0.3127,0.3290],\
                    colour.models.RGB_COLOURSPACE_BT2020.RGB_to_XYZ_matrix,\
                    chromatic_adaptation_transform='CAT02',\
                    cctf_decoding=colour.models.eotf_PQ_BT2100)
            lab = colour.XYZ_to_hdr_CIELab(xyz, illuminant=[ 0.3127, 0.329 ], Y_s=0.2, Y_abs=100, method='Fairchild 2011')

        chroma_feats = save_stats.chroma_feats(lab,C=color_C)

        Y_down = cv2.resize(Y,(dsize[1],dsize[0]),interpolation=cv2.INTER_CUBIC)

        gradient_x = cv2.Sobel(Y,ddepth=-1,dx=1,dy=0)
        gradient_y = cv2.Sobel(Y,ddepth=-1,dx=0,dy=1)
        gradient_mag = np.sqrt(gradient_x**2+gradient_y**2)    

        gradient_x_down = cv2.Sobel(Y_down,ddepth=-1,dx=1,dy=0)
        gradient_y_down = cv2.Sobel(Y_down,ddepth=-1,dx=0,dy=1)
        gradient_mag_down = np.sqrt(gradient_x_down**2+gradient_y_down**2)    


        Y_mscn,Ysigma,_ = save_stats.compute_image_mscn_transform(Y,C)
        dY_mscn,dYsigma,_ = save_stats.compute_image_mscn_transform(Y_down,C)

        gradY_mscn,_,_ = save_stats.compute_image_mscn_transform(gradient_mag,C)
        dgradY_mscn,_,_ = save_stats.compute_image_mscn_transform(gradient_mag_down,C)

        gradient_feats = save_stats.extract_secondord_feats(gradY_mscn)
        gdown_feats = save_stats.extract_secondord_feats(dgradY_mscn)
        gfeats = np.concatenate((gradient_feats,gdown_feats),axis=0)

        
        Ysigma_mscn,_,_= save_stats.compute_image_mscn_transform(Ysigma,C)
        dYsigma_mscn,_,_= save_stats.compute_image_mscn_transform(dYsigma,C)

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

            sts,sts_grad= find_kurtosis_sts(Y3d_mscn,grad3d_mscn,step,cy,cx,rst,rct,theta)
            dsts,dsts_grad= find_kurtosis_sts(Ydown_3d_mscn,graddown3d_mscn,step,dcy,dcx,rst,rct,theta)
            sts_arr = unblockshaped(np.reshape(sts,(-1,st_time_length,st_time_length)),r1*st_time_length,r2*st_time_length)
            sts_grad= unblockshaped(np.reshape(sts_grad,(-1,st_time_length,st_time_length)),r1*st_time_length,r2*st_time_length)

            dsts_arr = unblockshaped(np.reshape(dsts,(-1,st_time_length,st_time_length)),dr1*st_time_length,dr2*st_time_length)
            dsts_grad= unblockshaped(np.reshape(dsts_grad,(-1,st_time_length,st_time_length)),dr1*st_time_length,dr2*st_time_length)

            feats =  save_stats._extract_subband_feats(sts_arr)
            grad_feats = save_stats._extract_subband_feats(sts_grad)
            
            dfeats =  save_stats._extract_subband_feats(dsts_arr)
            dgrad_feats = save_stats._extract_subband_feats(dsts_grad)


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
    filename_out =os.path.join(os.path.splitext(name)[0]+'.z')
    joblib.dump(train_dict,os.path.join(results_folder,filename_out))
    return


def sts_fromvid(args):
    input_folder = './videos'
    filenames = glob.glob(os.path.join(input_folder,'*.yuv'))
    print(sorted(filenames))
    filenames = sorted(filenames)
    flag = 0
    os.makedirs(args.results_folder,exist_ok=True)
#    Parallel(n_jobs=15)(delayed(sts_fromfilename)(i,filenames,args.results_folder) for i in range(len(filenames)))
             



    return


def main():
    args = parser.parse_args()
    print(args)
#    sts_fromfilename(args.input_file,args.results_file,args.height,args.width,args.bit_depth,args.color_space)

if __name__ == '__main__':
    # print(__doc__)
    main()
    

