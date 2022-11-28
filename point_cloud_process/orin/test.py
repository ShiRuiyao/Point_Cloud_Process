import numpy as np
import laspy
from plyfile import PlyData, PlyElement
from pprint import pprint
import pandas as pd
import time
import os


def is_number(str):
    str = str.replace('.', '')
    str = str.replace(' ', '')
    str.strip()
    if str.isnumeric():
        return True
    else:
        return False

x=1
y=2.2
z=False
dict = {'x': x, 'y': y, 'z': z}
try:
    a=dict['hi']
    print(a)
except(KeyError):
    print('KeyError')

if 'a' in locals().keys():
    print('you')
else:
    print('meiyou')
# npy_file='/home/utils/datadisk/Hessigheim_Benchmark/Epoch_March2018/train_test_split/train/Mar18_val.npy'
# time_start = time.time()  # 记录开始时间
# pc_array = pd.read_csv(npy_file, sep='\s+', header=[0],error_bad_lines=False).values
# time_end = time.time()  # 记录结束时间
# time_sum = time_end - time_start
# print(time_sum)
# pc_array = np.load(npy_file)

# las_file='/home/utils/datadisk/Hessigheim_Benchmark/Epoch_March2016/LiDAR/Mar16_val.laz'
# las = laspy.read(las_file)
# i=0
# try:
#     red=las.classification
#     i+=1
# except (AttributeError):
#     print('meiyou1')
# array12 = np.bincount(red) # 获得数组中每个值出现的个数

# test=np.array([1,1,1,2,2,3])
# test2=np.vstack((test,test))
# a=test2[2,2]
# np.savetxt('1.txt',test)
# arrayc = np.bincount(test)
# filename='/home/utils/datadisk/SensatUrban/ply/train/birmingham_block_0.ply'
# a=os.path.basename(filename)[:-len('npy')]
# plydata = PlyData.read(filename)
# with laspy.open(las_file) as fh:
#     print('Points from Header:', fh.header.point_count)
#     las = fh.read()
# clouds=np.array(las.xyz)
# format_nums=len(las.point_format.dimensions)
# colors=np.vstack((las.red,las.green,las.blue)).T.astype(float)
# colorsmax=np.max(colors,axis=0)
# colorsmin=np.min(colors,axis=0)
# print('RGB最大值为：',colorsmax)
# print()
# colors=(colors-colorsmin)*255/60160
# PC=np.hstack((clouds,colors))
# np.savetxt('pc.txt',PC)
# str = '12321313.123123123'
# # print(is_number(str))
#
# txt_file = '/home/yym/lhm/Hessigheim_Benchmark/Epoch_March2016/result/Mar16_test.txt'
# # npy_file='/home/yym/lhm/Hessigheim_Benchmark/Epoch_March2016/fullarea/Mar16_test.npy'
# # npy_file='/home/yym/lhm/PAConv-main/scene_seg/data/s3dis_orin/trainval_fullarea/Area_1_conferenceRoom_1.npy'
# npy_file = '/home/yym/lhm/Hessigheim_Benchmark/Epoch_March2018/train_test_split/test/Mar18_test_48.npy'
# # clouds = np.loadtxt(txt_file)
# clouds = np.load(npy_file)
# cloud = clouds[:, 0:3]
# maxc = np.max(cloud, 0)
# minc = np.min(cloud, 0)
# x=np.zeros((3,4))
# y=[1,1,1]
# x1=x[:,0]
# x1+=y[0]


print('hello')
