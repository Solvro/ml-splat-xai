import unittest
import torch
from pathlib import Path
#import taichi as ti
from pointnet import PointNet2ClsSSG, PointNet2ClsMSG, PointNet2SegMSG, PointNet2SegSSG, PointNet2PartSegSSG, \
    PointNet2PartSegMSG
from pointnet.dataset import GaussianPointCloud

#ti.init(ti.cuda)
DATA_PATH = Path('./data/train')

class Test(unittest.TestCase):

    def test_pointnet2_cls_ssg(self):
        x = torch.randn(2, 3, 1024).cuda()
        xyz = x.clone()
        model = PointNet2ClsSSG(3, 40).cuda()
        out = model(x, xyz)
        assert out.shape == (2, 40)

    def test_pointnet2_cls_msg(self):
        x = torch.randn(2, 3, 1024).cuda()
        xyz = x.clone()
        model = PointNet2ClsMSG(3, 40).cuda()
        out = model(x, xyz)
        assert out.shape == (2, 40)

    def test_pointnet2_seg_ssg(self):
        x = torch.randn(2, 3, 1024).cuda()
        xyz = x.clone()
        model = PointNet2SegSSG(3, 40).cuda()
        out = model(x, xyz)
        assert out.shape == (2, 40, 1024)

    def test_pointnet2_seg_msg(self):
        x = torch.randn(2, 3, 1024).cuda()
        xyz = x.clone()
        model = PointNet2SegMSG(3, 40).cuda()
        out = model(x, xyz)
        assert out.shape == (2, 40, 1024)

    def test_pointnet2_part_seg_ssg(self):
        x = torch.randn(2, 3, 1024).cuda()
        xyz = x.clone()
        category = torch.randint(0, 10, (2,)).cuda()
        model = PointNet2PartSegSSG(3, 40).cuda()
        out = model(x, xyz, category)
        assert out.shape == (2, 40, 1024)

    def test_pointnet2_part_seg_msg(self):
        x = torch.randn(2, 3, 1024).cuda()
        xyz = x.clone()
        category = torch.randint(0, 10, (2,)).cuda()
        model = PointNet2PartSegMSG(3, 40).cuda()
        out = model(x, xyz, category)
        assert out.shape == (2, 40, 1024)

    def test_dataset_random_seed_equal(self):
        dataset_random1 = GaussianPointCloud(DATA_PATH, num_points=2048, sampling_method="random", random_seed=42)
        dataset_random2 = GaussianPointCloud(DATA_PATH, num_points=2048, sampling_method="random", random_seed=42)

        assert torch.equal(dataset_random1[0]['xyz'], dataset_random2[0]['xyz']), "Random sampling with the same seed should yield the same results"

    def test_dataset_random_seed_not_equal(self):
        dataset_fps1 = GaussianPointCloud(DATA_PATH, num_points=2048, sampling_method="random", random_seed=88)
        dataset_fps2 = GaussianPointCloud(DATA_PATH, num_points=2048, sampling_method="random", random_seed=42)
        assert not torch.equal(dataset_fps1[0]['xyz'], dataset_fps2[0]['xyz']), "Farthest point sampling with the different seed should yield different results"

    def test_dataset_fps_equal(self):
        dataset_fps1 = GaussianPointCloud(DATA_PATH, num_points=2048, sampling_method="fps", random_seed=42)
        dataset_fps2 = GaussianPointCloud(DATA_PATH, num_points=2048, sampling_method="fps", random_seed=42)
        assert torch.equal(dataset_fps1[0]['xyz'], dataset_fps2[0]['xyz']), "Farthest point sampling with the same seed should yield the same results"

    def test_dataset_original_size_equal(self):
        dataset_original1 = GaussianPointCloud(DATA_PATH, num_points=2048, sampling_method="original_size")
        dataset_original2 = GaussianPointCloud(DATA_PATH, num_points=2048, sampling_method="original_size")
        assert torch.equal(dataset_original1[0]['xyz'], dataset_original2[0]['xyz']), "Original size sampling should yield the same results"


if __name__ == '__main__':
    unittest.main()