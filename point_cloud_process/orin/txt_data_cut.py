import os
from tqdm import tqdm
import time
import shutil
from utils import filepath

txt_file = 'F:/datasets/Hessigheim_Benchmark/Epoch_March2018/LiDAR/Mar18_test.txt'
# txt文件路径
file_dir = txt_file.split('/')[0] + '/' + txt_file.split('/')[1] + '/' + txt_file.split('/')[2] + '/' + \
           txt_file.split('/')[3] + '/' + 'result' + '/'
# 设定移动目录
result_dir = 'E:' + '/' + txt_file.split('/')[1] + '/' + txt_file.split('/')[2] + '/' + txt_file.split('/')[
    3] + '/' + 'result' + '/'


# 设定输出目录，由于移动硬盘写入较慢改为本机固态写入而后移动至数据盘

def readline_count(f):
    return len(open(f).readlines())


def txt_cut(txt_file, file_dir, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = os.path.basename(txt_file)
    with open(txt_file, "r") as f:
        list2 = []
        data = f.readlines()
        num_lines = readline_count(txt_file)
        print("总行数为", num_lines)
        start = time.perf_counter()
        for i in tqdm(range(len(data))):
            hang = data[i]
            if filepath.is_number(hang) == False:
                continue
            list = []
            for x in hang.strip().split():
                list.append(x)
            del list[6:9]
            s = '\t'.join(list) + '\n'
            with open(result_dir + file_name, 'a+') as q:
                q.write(s)
    filepath.mymovefile(result_dir + file_name, file_dir)


def txt_reprocess(txt_file, result_dir):
    file_name = os.path.basename(txt_file)
    with open(txt_file, "r") as f:
        list2 = []
        data = f.readlines()
        num_lines = readline_count(txt_file)
        print("总行数为", num_lines)
        start = time.perf_counter()
        for i in tqdm(range(len(data))):
            hang = data[i]
            list = []
            for x in hang.strip().split():
                list.append(x)
            if filepath.is_number(x) == False:
                continue
            s = hang
            with open(result_dir + file_name, 'a+') as q:
                q.write(s)


if __name__ == '__main__':
    # txt_cut(txt_file=txt_file,file_dir=file_dir,result_dir=result_dir)
    txt_reprocess('/home/yym/lhm/Hessigheim_Benchmark/Epoch_March2018/result/Mar18_test.txt','/home/yym/lhm/Hessigheim_Benchmark/Epoch_March2018/')
