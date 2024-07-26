import torch
from torch.utils.data import Dataset
import open3d as o3d
from pathlib import Path
import numpy as np

class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir : Path, n_sample : int, class_list : list[str], is_training : bool):
        self.root_dir : Path = Path(root_dir)
        self.n_sample : int = n_sample
        self.class_list : int = class_list
        self.isTraining = is_training
        self.point_cloud = []
        self.label = []

        self.load_data()

    def __len__(self):
        assert len(self.point_cloud) == len(self.label)
        return len(self.point_cloud)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.point_cloud[idx].points) # (n_points, 3)
        y = torch.FloatTensor(self.label[idx]) # (label, )
        return x,y

    def load_data(self):
        class_dir_list = [class_dir for class_dir in self.root_dir.iterdir()]
        
        for class_dir in class_dir_list:
            if class_dir.name not in self.class_list:
                continue

            train_dir : Path = Path()
            test_dir : Path = Path()
            for sub_dir in class_dir.iterdir():
                if sub_dir.name == "train":
                    train_dir = sub_dir
                elif sub_dir.name == "test":
                    test_dir = sub_dir
                else:
                    assert False

            if self.isTraining == True:
                for file_path in train_dir.iterdir():
                    mesh = o3d.io.read_triangle_mesh(file_path.absolute().__str__())
                    if mesh.is_empty():
                        self.ensure_off_newline(file_path)

                    pcd = mesh.sample_points_uniformly(number_of_points=self.n_sample)
                    self.point_cloud.append(pcd)
                    self.label.append(self.class_to_onehot(class_dir.name))
            else:
                for file_path in test_dir.iterdir():
                    mesh = o3d.io.read_triangle_mesh(file_path.absolute().__str__())
                    if mesh.is_empty():
                        self.ensure_off_newline(file_path)

                    pcd = mesh.sample_points_uniformly(number_of_points=self.n_sample)
                    self.point_cloud.append(pcd)
                    self.label.append(self.class_to_onehot(class_dir.name))

    def class_to_onehot(self, label : str):
        index = self.class_list.index(label)
        label_onehot = torch.zeros(len(self.class_list))
        label_onehot[index] = 1

        return label_onehot

    def class_to_index(self, label : str):
        return self.class_list.index(label)

    def ensure_off_newline(self, file_path):
        # 파일을 읽기 모드로 열기
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 첫 번째 라인을 확인하고 "OFF"가 포함된 경우 줄 바꿈 삽입
        first_line = lines[0]
        if "OFF" != first_line:
            # "OFF"를 찾아 줄 바꿈 삽입
            index = first_line.index("OFF") + 3
            first_line = first_line[:index] + '\n' + first_line[index:]
            lines[0] = first_line
        else:
            raise ValueError("[ERROR] Check if the file is valid.")

        # 파일을 쓰기 모드로 열기
        with open(file_path, 'w') as file:
            file.writelines(lines)


    def zero_mean(self):
        for points in self.point_cloud:
            center = points.get_center()
            points.translate(-center)

    def scale(self):
        for points in self.point_cloud:
            min_bound = points.get_min_bound()
            max_bound = points.get_max_bound()
            scale_factor : np.ndarray = max_bound - min_bound
            points.points = o3d.utility.Vector3dVector(((points.points - min_bound) / scale_factor) * 2 - 1.0)

    def normalize(self):
        self.zero_mean()
        self.scale()