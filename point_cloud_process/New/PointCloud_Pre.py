import os
import numpy as np
import laspy
import h5py
from helper_ply import write_ply,read_ply
from plyfile import PlyData, PlyElement
import tqdm
import time
from sklearn.neighbors import KDTree
import pickle
import nearest_neighbors
import grid_subsampling


class PCdata_trans:
    """
    run PCdata_trans.main to trans your data to type you want
    """

    def __init__(self, src_file_path, dis_file_path, dis_file_type, src_file_type=None,split='train'):
        self.train_files = []
        self.test_files = []
        self.val_files = []
        self.src_file_path = src_file_path
        self.dis_file_path = dis_file_path
        self.src_file_type = src_file_type
        self.dis_file_type = dis_file_type
        self.split=split

    def main(self):
        self.src_filelist = os.listdir(self.src_file_path)
        if not self.src_file_type:
            self.src_file_type = os.path.splitext(self.src_filelist[0])[-1]
        num_per_class = []
        for src_file in self.src_filelist:
            print('Processing point cloud data files：', os.path.basename(src_file))
            src_file = self.src_file_path + '/' + os.path.basename(src_file)
            pc_array = self.getclouds(src_file)
            if self.split == 'train':
                num_per_class.append(np.bincount(pc_array[:, -1].astype(int)))
            dis_file = self.dis_file_path + '/' + os.path.basename(src_file)[:-len(self.src_file_type)] + '.'
            self.writeclouds(pc_array=pc_array, dis_file=dis_file)
        # nums_per_class = np.zeros_like(num_per_class[0])
        if self.split == 'train':
            num_of_classes = 0
            for i in range(len(num_per_class)):
                if len(num_per_class[i]) > num_of_classes:
                    num_of_classes = len(num_per_class[i])
                else:
                    pass
            for i in range(len(num_per_class)):
                if len(num_per_class[i]) < num_of_classes:
                    num_per_class[i] = np.pad(num_per_class[i], (0, num_of_classes - len(num_per_class[i]),), 'constant')
                else:
                    pass
            nums_per_class = np.sum(num_per_class, axis=0)
        np.save(self.dis_file_path + '/nums_per_class.npy', nums_per_class)

    def getclouds(self, src_file):
        """ Using numpy as intermediate storage medium """
        print('Loading point cloud data files：', os.path.basename(src_file))
        if self.src_file_type == '.ply':
            pc_array = self.getpoints_ply(src_file)

        elif self.src_file_type == '.npy':
            pc_array = np.load(src_file)

        elif self.src_file_type == '.txt':
            pc_array = np.loadtxt(src_file)

        elif self.src_file_type == '.las' or self.src_file_type == '.laz':
            pc_array = self.getpoints_las(src_file)

        return pc_array

    def getpoints_ply(self, src_file):
        """ read XYZ point cloud and format from filename PLY file """
        ply_data=read_ply((src_file))
        try:
            pc_array=np.vstack((ply_data['x'],ply_data['y'],ply_data['z'],ply_data['red'],ply_data['green'],ply_data['blue'],ply_data['class'])).T
        except(ValueError):
            pc_array = np.vstack(
                (ply_data['x'], ply_data['y'], ply_data['z'], ply_data['red'], ply_data['green'], ply_data['blue'])).T
        # plydata = PlyData.read(src_file)
        # pc = plydata['vertex'].data
        # len_pc = len(pc)
        # len_pc_pick = len(pc[0])
        # print('开始读取ply文件：', src_file)
        # if len_pc_pick == 7:
        #     pc_array = np.array([[x, y, z, r, g, b, c] for x, y, z, r, g, b, c in pc])
        # elif len_pc_pick == 6:
        #     pc_array = np.array([[x, y, z, r, g, b] for x, y, z, r, g, b in pc])
        # elif len_pc_pick == 4:
        #     pc_array = np.array([[x, y, z, c] for x, y, z, c in pc])
        # elif len_pc_pick == 3:
        #     pc_array = np.array([[x, y, z] for x, y, z in pc])
        # pc_shape = np.shape(pc_array)
        return pc_array

    def getpoints_las(self, src_file):
        clouds = []
        with laspy.open(src_file) as fh:
            print('Points from Header:', fh.header.point_count)
            las = fh.read()
        clouds.append(np.array(las.xyz))
        try:
            clouds.append(np.array(las.red))
            clouds.append(np.array(las.green))
            clouds.append(np.array(las.blue))
        except (AttributeError):
            print('There is no color data this file.')
        try:
            clouds.append(np.array(las.classification))
        except (AttributeError):
            print('There is no classification data this file.')

        format_num = len(clouds)
        pc_array = clouds[0]
        for j in range(format_num - 1):
            pc_array = np.hstack((pc_array, clouds[j + 1].reshape(-1,1)))
        return pc_array

    def getpoints_h5(self, src_file):  # unfinished
        f = h5py.File(src_file, 'r')

    def writeclouds(self, pc_array, dis_file):
        if self.dis_file_type == 'ply':
            self.write_ply(pc_array, dis_file + self.dis_file_type)
        elif self.dis_file_type == 'npy':
            np.save(dis_file + self.dis_file_type, pc_array)
        elif self.dis_file_type == 'txt':
            np.savetxt(dis_file + self.dis_file_type, pc_array)
        elif self.dis_file_type == 'las' or self.dis_file_type == 'laz':
            self.write_las(pc_array, dis_file + self.dis_file_type)

    def write_ply(self, pc_array, dis_file, text=True):
        """
            dis_file : path to save: '/yy/XX.ply'
            pc_array: size (N,3) or (N,4) or (N,6) or (N,7)
        """
        len_pc_pick = len(pc_array[0])
        print('Reading the data written to the ply file：')
        coords = [(pc_array[i, 0], pc_array[i, 1], pc_array[i, 2]) for i in range(pc_array.shape[0])]
        j = 3
        el = PlyElement.describe(np.array(coords, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex', comments=['coordinates'])
        try:
            color = [(pc_array[i, j], pc_array[i, j + 1], pc_array[i, j + 2]) for i in range(pc_array.shape[0])]
            j += 3
            colors = PlyElement.describe(np.array(color, dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'color', comments=['colors'])
        except(IndexError):
            print('Input point cloud without color information.')
        try:
            classification = [(pc_array[i, j]) for i in range(pc_array.shape[0])]
            j += 1
            classifications = PlyElement.describe(np.array(classification, dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'color',
                                                  comments=['labels'])
        except(IndexError):
            print('Input point cloud without classification information.')
        print('开始写入~')
        if j == 3:
            PlyData([el], text=text).write(dis_file)
        elif j == 4:
            PlyData([el, classifications], text=text).write(dis_file)
        elif j == 6:
            PlyData([el, colors], text=text).write(dis_file)
        elif j == 7:
            PlyData([el, colors, classifications], text=text).write(dis_file)
        print('写入完成啦~')

    def write_las(self, pc_array, dis_file):
        # 1. Create a new header
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))
        header.offsets = np.min(pc_array, axis=0)
        header.scales = np.array([0.1, 0.1, 0.1])

        # 2. Create a Las
        las = laspy.LasData(header)

        las.x = pc_array[:, 0]
        las.y = pc_array[:, 1]
        las.z = pc_array[:, 2]
        las.red = pc_array[:, 3]
        las.green = pc_array[:, 4]
        las.blue = pc_array[:, 5]
        las.random = np.random.randint(-1503, 6546, len(las.points), np.int32)

        las.write(dis_file)

    def write_h5(self, fname, pc):  # unfinished
        fp = h5py.File(fname, 'w')
        coords = pc[:, 0:3]
        points = pc[:, 4:7]
        labels = pc[:, 7:8]
        points = np.hstack((coords, points))
        fp.create_dataset('data', data=points, compression='gzip', dtype='float32')
        fp.create_dataset('label', data=labels, compression='gzip', dtype='int64')
        fp.close()
        print('saved:', fname)


def sub_and_pkl(coords, colors, labels, dis_file, sub_grid_size=0.3):
    """
    :param coords: xyz
    :param colors: RGB
    :param labels: classification
    :param dis_file: a file path without filename extension but have '.' eg:/home/utils/datadisk/SensatUrban/ply/train/birmingham_block_0.
    :param sub_grid_size:
    """
    sub_xyz, sub_colors, sub_labels = DataProcessing.grid_sub_sampling(coords, colors, labels, sub_grid_size)
    sub_colors = sub_colors / 255.0
    sub_ply_file = dis_file.replace('.', 'sub_sampling.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    search_tree = KDTree(sub_xyz)

    kd_tree_file = dis_file.replace('.', '_KDTree.pkl')

    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(coords, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)

    proj_save = dis_file.replace('.', '_proj.pkl')

    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)


class DataProcessing:

    @staticmethod
    def get_file_list(dataset_path, test_scan_num):
        seq_list = np.sort(os.listdir(dataset_path))

        train_file_list = []
        test_file_list = []
        val_file_list = []
        for seq_id in seq_list:
            seq_path = os.path.join(dataset_path, seq_id)
            pc_path = os.path.join(seq_path, 'velodyne')
            if seq_id == '08':
                val_file_list.append([os.path.join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
                if seq_id == test_scan_num:
                    test_file_list.append([os.path.join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif int(seq_id) >= 11 and seq_id == test_scan_num:
                test_file_list.append([os.path.join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                train_file_list.append([os.path.join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        train_file_list = np.concatenate(train_file_list, axis=0)
        val_file_list = np.concatenate(val_file_list, axis=0)
        test_file_list = np.concatenate(test_file_list, axis=0)
        return train_file_list, val_file_list, test_file_list

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return grid_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return grid_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return grid_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return grid_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                            verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(dataset_name):
        # pre-calculate the number of points in each category
        num_per_class = []
        if dataset_name is 'STPLS3D':
            num_per_class = np.array([5315782, 6445416, 3669377, 170255, 31557, 89423], dtype=np.int32)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)


def is_number(str):
    str = str.replace('.', '')
    str = str.replace(' ', '')
    str.strip()
    if str.isnumeric():
        return True
    else:
        return False


def readline_count(f):
    return len(open(f).readlines())


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
            if is_number(x) == False:
                continue
            s = hang
            with open(result_dir + file_name, 'a+') as q:
                q.write(s)
