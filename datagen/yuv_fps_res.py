import pandas as pd
import os

csvfile = 'fall2021_hdr_res_fps_fnos.csv'
df = pd.read_csv(csvfile,names=['names','res','fps','framenos'])

names=df["names"]
fps_list = df["fps"] 
res = df["res"]
framenos_list = df["framenos"]
yuv_names = []
fps_nlist = []
w_list = []
h_list = []
framenos_nlist = []

for i,name in enumerate(names):
    basename = os.path.basename(name)
    w_list.append(int(res[i].split('x')[0]))
    h_list.append(int(res[i].split('x')[1]))
    num,denom = fps_list[i].split('/')
    fps_nlist.append(float(num)/float(denom))
    framenos_nlist.append(framenos_list[i])
    yuv_names.append(basename[:-3]+'yuv')
df = pd.DataFrame(list(zip(yuv_names,framenos_nlist,fps_nlist,w_list,h_list)),columns=['yuv','framenos','fps','w','h'])

df.to_csv('~/fall2021_yuv_rw_info.csv')
