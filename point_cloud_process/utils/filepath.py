import os
import shutil
import numpy as np
import math
from tqdm import tqdm
import random


def is_number(str):
    str = str.replace('.', '')
    str = str.replace(' ', '')
    str.strip()
    if str.isnumeric():
        return True
    else:
        return False


def aseert_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def mymovefile(srcfile, dstpath):
    # 将srcfile复制到dstpath目录
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.move(srcfile, dstpath + fname)  # 复制文件
        print("move %s -> %s" % (srcfile, dstpath + fname))


def npy_cut(src_data_path, dis_path):
    src_files = os.listdir(src_data_path)
    src_data_files = [src_data_path + '/' + x for x in src_files]
    for i in src_data_files:
        cloud = np.load(i)
        num_split = int(cloud.shape[0] / 81920)
        clouds = np.zeros((num_split + 1, 81920, cloud.shape[1]))
        aseert_mkdir(dis_path)
        for j in tqdm(range(num_split)):
            if j < num_split:
                clouds[j, :, :] = cloud[j * 81920:(j + 1) * 81920, :]
            elif j == num_split:
                clouds[j, :, :] = cloud[-80920:, :]
            single_f_name = i.split('/')[-1][:-4] + '_' + str(j) + '.npy'
            np.save(dis_path + '/' + single_f_name, clouds[j, :, :])

def filename_area_add(src_data_path,area_nums):
    src_files = os.listdir(src_data_path)
    src_data_files = [src_data_path + '/' + x for x in src_files]
    for i in src_data_files:
        randarea=area_nums[random.randint(0, len(area_nums) - 1)]
        filename_add=src_data_path+'/Area_'+str(randarea)+'_'+i.split('/')[-1]
        os.rename(i,filename_add)



if __name__ == '__main__':
    area_nums=[1]
    filename_area_add('/home/yym/lhm/Hessigheim_Benchmark/Epoch_March2016/fullarea_7',area_nums)
    print('Hi')
