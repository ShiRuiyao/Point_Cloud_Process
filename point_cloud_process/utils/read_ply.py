import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    len_pc = len(pc)
    len_pc_pick = len(pc[0])
    print('开始读取ply文件：')
    if len_pc_pick == 6:
        pc_array = np.array([[x, y, z, r, g, b] for x, y, z, r, g, b in tqdm(pc)])
    elif len_pc_pick == 3:
        pc_array = np.array([[x, y, z] for x, y, z in tqdm(pc)])
    pc_shape = np.shape(pc_array)
    return pc_array, pc_shape


def write_ply_color(points, filename, text=True):
    """ input: Nx6, write points to filename as PLY format. """
    points_ = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    points_color_ = [(255 * points[i, 3], 255 * points[i, 4], 255 * points[i, 5]) for i in range(points.shape[0])]

    n = points.shape[0]
    vertex = np.array(points_, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_color = np.array(points_color_, dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_all = np.empty(n, vertex.dtype.descr + vertex_color.dtype.descr)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    for prop in vertex_color.dtype.names:
        vertex_all[prop] = vertex_color[prop]

    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)


def write_ply(save_path, points, text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3) or (N,6)
    """
    len_pc_pick = len(points[0])
    print('正在读取写入ply文件的数据：')
    if len_pc_pick == 3:
        points = [(points[i, 0], points[i, 1], points[i, 2]) for i in tqdm(range(points.shape[0]))]
    elif len_pc_pick == 6:
        points = [(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4], points[i, 5]) for i in
                  tqdm(range(points.shape[0]))]
    vertex = np.array(points,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    print('开始写入~')
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)
    print('写入完成啦~')


if __name__ == '__main__':
    ply_path1 = 'F:/datasets/PC/Swiss3DCities_Aerial_Photogrammetric_3D_Pointcloud_Dataset_with_Semantic_Labels/Medium/Medium/1_Davos_16_34556_-23105/1_terrain.ply'
    ply_path2 = 'F:/datasets/PC/Swiss3DCities_Aerial_Photogrammetric_3D_Pointcloud_Dataset_with_Semantic_Labels/remake/1_terrain.ply'
    save_path = 'F:/datasets/PC/Swiss3DCities_Aerial_Photogrammetric_3D_Pointcloud_Dataset_with_Semantic_Labels/remake/1_terrain.ply'
    # pcarray1, _ = read_ply(ply_path1)
    # write_ply(save_path, pcarray1)
    pcarray2, _ = read_ply(ply_path2)
