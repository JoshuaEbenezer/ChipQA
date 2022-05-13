from mpi4py import MPI
from joblib import load
import argparse
import glob
import os

parser = argparse.ArgumentParser(description='Generate HDR BRISQUE features from a folder of videos and store them')
parser.add_argument('--input_folder',help='Folder containing input videos')
parser.add_argument('--results_folder',help='Folder where features are stored')

args = parser.parse_args()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

orig_files = glob.glob(os.path.join(args.input_folder,'*.yuv'))
files = []
for vname in orig_files:
    results_folder = args.results_folder
    name = os.path.basename(vname)
    filename_out =os.path.join(results_folder,os.path.splitext(name)[0]+'.z')
    if(os.path.exists(filename_out)==True):
        continue    
    files.append(vname)


for i in range(rank, len(files), size):
    vname = files[i]
    results_folder = args.results_folder
    name = os.path.basename(vname)
    os.makedirs(results_folder,exist_ok=True)
    results_file = os.path.join(results_folder,os.path.splitext(os.path.basename(vname))[0]+'.z')
    cmd = "python3 ../chipqa_yuv.py --input_file {vname} --results_file {results_file} --bit_depth 10 --color_space BT2020 --width 3840 --height 2160".format(vname=vname,results_file=results_file)
    print(cmd)
    os.system(cmd)
