from PointCloud_Pre import *
import numpy as np

if __name__ == '__main__':
    # data = PCdata_trans('', '', '', 'npy')
    # clouds, numsperclass = [], []
    # # clouds.append(data.getclouds('/home/utils/datadisk/Hessigheim_Benchmark/Epoch_March2018/train_test_split/train/Mar18_train.npy')[:, -1])
    # # # clouds.append(data.getclouds('/home/utils/datadisk/Hessigheim_Benchmark/Epoch_March2018/train_test_split/train/Mar18_val.npy')[:, -1])
    # # # # clouds0=clouds0.astype(int)
    # # # clouds1 = clouds1.astype(int)
    # numsperclass = []
    # file_path = '/home/utils/datadisk/SensatUrban/npy/train'
    # file_list = os.listdir(file_path)
    # for i in file_list:
    #     print('load file:', i)
    #     cloud = np.load(file_path + '/' + i)
    #     numsperclass.append(np.bincount(cloud[:, -1].astype(int)))
    # classes = 0
    # for i in range(len(numsperclass)):
    #     if len(numsperclass[i]) > classes:
    #         classes = len(numsperclass[i])
    #     else:
    #         pass
    # for i in range(len(numsperclass)):
    #     if len(numsperclass[i]) < classes:
    #         numsperclass[i] = np.pad(numsperclass[i], (0, classes - len(numsperclass[i])), 'constant')
    #     else:
    #         pass
    # numperclass = np.sum(numsperclass,axis=0)
    # print(numperclass)
    # # numsperclass.append(np.bincount(clouds[1].astype(int)))
    # # numperclass = np.add(numsperclass[0], numsperclass[1])
    # # print(numperclass)
    # # print('hi')
    # data_file1 = '/home/utils/datadisk/Hessigheim_Benchmark/Epoch_March2016/fullarea_7/Area_1_Mar16_train.npy'
    # data_file2 = '/home/utils/datadisk/Hessigheim_Benchmark/Epoch_March2016/fullarea_7/Area_1_Mar16_val.npy'
    # pc_array,numperclass=[],[]
    # pc_array.append(np.load(data_file1))
    # pc_array.append(np.load(data_file2))
    # for i in pc_array:
    #     numperclass.append(np.bincount(i[:,-1].astype(int)))
    # for j in numperclass:
    #     numsperclass=np.sum(numperclass,axis=0)
    # print(numsperclass)
    # print('hi')

    src_file_path = '/home/utils/datadisk/Hessigheim_Benchmark/Epoch_March2018/processed_data/LiDAR_cut'
    dis_file_path = '/home/utils/datadisk/Hessigheim_Benchmark/Epoch_March2018/processed_data/npy'
    data = PCdata_trans(src_file_path, dis_file_path,'npy')
    data.main()
