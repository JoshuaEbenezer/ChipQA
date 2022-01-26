import os
import subprocess
import pandas as pd
import glob
from joblib import Parallel, delayed


metadata_csv = pd.read_csv("./hash_list.csv")

def expand_res_name(res_shorthand):
    if(res_shorthand=='2160p'):
        resolution='3840x2160'
    elif(res_shorthand=='1080p'):
        resolution='1920x1080'
    elif(res_shorthand=='720p'):
        resolution='1280x720'
    elif(res_shorthand == '540p'):
        resolution='960x540'
    return resolution
def upscale_single_vid(full_name,out_name):
    cmd = ['/script_upscale_y4m.sh', full_name,out_name]
    print(cmd)
    p = subprocess.Popen(['./script_upscale_y4m.sh', full_name,out_name])
    p.wait()
    return

def run_ffmpeg(metadata_csv,index):
    row = metadata_csv.iloc[index]
    name = row['original_video_name']
    if(name!='EPLNight_SRC_SRC_SRC_SRC_0.y4m'):
        return 
    full_name = os.path.join('/data/PV_VQA_Study/all_cut_y4m_vids',name)
    print(full_name)
    out_name = os.path.join('/data/PV_VQA_Study/all_cut_upscaled_y4m_vids/',name)
    cmd = ['/script_upscale_y4m.sh', full_name,out_name]
    print(cmd)
    p = subprocess.Popen(['./script_upscale_y4m.sh', full_name,out_name])
    p.wait()
    return

#Parallel(n_jobs=32)(delayed(run_ffmpeg)(metadata_csv,i) for i in range(0,len(metadata_csv)))
for i in range(len(metadata_csv)):
    run_ffmpeg(metadata_csv,i)
