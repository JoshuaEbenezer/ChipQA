import numpy as np

def fread(fid, nelements, dtype):
     if dtype is np.str:
         dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
     else:
         dt = dtype

     data_array = np.fromfile(fid, dt, nelements)
     data_array.shape = (nelements, 1)

     return data_array




def yuv_read(filename,frame_num,height,width,bit_depth):
    file_object = open(filename)
    if(bit_depth==8):
        file_object.seek(frame_num*height*width*1.5)
        y1 = fread(file_object,height*width,np.uint8)
        u1 = fread(file_object,height*width//4,np.uint8)
        v1 = fread(file_object,height*width//4,np.uint8)
    elif(bit_depth==10 or bit_depth==12):
        file_object.seek(frame_num*height*width*3)
        y1 = fread(file_object,height*width,np.uint16)
        u1 = fread(file_object,height*width//4,np.uint16)
        v1 = fread(file_object,height*width//4,np.uint16)
    y = np.reshape(y1,(height,width))
    u = np.reshape(u1,(height//2,width//2)).repeat(2,axis=0).repeat(2,axis=1)
    v = np.reshape(v1,(height//2,width//2)).repeat(2,axis=0).repeat(2,axis=1)
    return y,u,v

def yuv2rgb_bt2020(y,u,v):
    # cast to float32 for yuv2rgb in BT2020
    y = y.astype(np.float32)
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    cb = u - 512
    cr = v - 512

    r = y+1.4747*cr
    g = y-0.1645*cb-0.5719*cr
    b = y+1.8814*cb

    r = r.astype(np.uint16)
    g = g.astype(np.uint16)
    b = b.astype(np.uint16)

    frame = np.stack((r,g,b),2)
    return frame

