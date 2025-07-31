from pointnets.pointnet2 import PointNet2ClsMSG, PointNet2ClsSSG

model_list = {
    "pointnet2_ssg": (PointNet2ClsSSG, None, 1024),
    "pointnet2_msg": (PointNet2ClsMSG, None, 1280),
}
