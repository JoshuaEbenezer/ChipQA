# python script to increase fps of distorted videos to the same FR as original
from joblib import Parallel,delayed

import subprocess
import os
import pandas as pd
import glob

def fps_from_content(content,fr):
    if(content=='EPLDay' or content=='EPLNight' or content=='Cricket1' or content=='Cricket2' or content=='USOpen'):
        if(fr=='HFR'):
            fps = 50
        else:
            fps = 25
    elif(content=='TNFF' or content=='TNFNFL'):
        if(fr=='HFR'):
            fps = 59.94
        else:
            fps = 29.97
    return fps

def expand_res_name(res_shorthand):
    print(res_shorthand)
    if(res_shorthand=='720p'):
        resolution='1280x720'
    elif(res_shorthand == '540p'):
        resolution='960x540'
    elif(res_shorthand == '396p'):
        resolution='704x396'
    elif(res_shorthand == '288p'):
        resolution='512x288'
    return resolution

metadata_csv = pd.read_csv('./lbvfr_distorted_metadata.csv')
frame_duplicate = False

def run_ffmpeg(metadata_csv,index):
    row = metadata_csv.iloc[index]
    name = row['original_video_name']
    content = row['content']
    hfr_fps = str(fps_from_content(content,'HFR'))
    fr = name.split('_')[2]
    full_name = os.path.join('/data/PV_VQA_Study/all_cut_upscaled_y4m_vids',name)
    print(full_name)
    if(fr=='HFR'):
        out_name = os.path.join('/data/PV_VQA_Study/all_cut_upscaled_hfr_motioninterpolated_yuv_vids',os.path.splitext(name)[0]+'.yuv')
        if(os.path.exists(out_name)):
            os.remove(out_name)
        print('already HFR')
        return

    if(frame_duplicate==True):
        out_name = os.path.join('/data/PV_VQA_Study/all_cut_upscaled_hfr_yuv_vids',os.path.splitext(name)[0]+'.yuv')
        p = subprocess.Popen(['./script_increase_fps.sh', full_name,out_name,hfr_fps])
    else: # motion interpolation
        out_name = os.path.join('/data/PV_VQA_Study/all_cut_upscaled_hfr_motioninterpolated_yuv_vids',os.path.splitext(name)[0]+'.yuv')
        print('to be done')
        if(os.path.exists(out_name)==True):
            print('already exists')
            return
        p = os.system(' '.join(['./script_mint_increase_fps.sh', full_name,out_name,hfr_fps]))
    return 

Parallel(n_jobs=30)(delayed(run_ffmpeg)(metadata_csv,i) for i in range(0,len(metadata_csv)))
#for p in P:
#    p.wait()
#for i in range(len(metadata_csv)):
#    run_ffmpeg(metadata_csv,i)
