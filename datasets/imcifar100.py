import os
import numpy as np
from .utils import DatasetBase, Datum
from torchvision.datasets import CIFAR100
from .template import template


class IMBALANCECIFAR100(DatasetBase):
    """IMBALANCE CIFAR-10 dataset class as a subclass of DatasetBase."""

    dataset_dir = 'cifar-10-imbalance'

    def __init__(self, root, imbalance_ratio=50):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.imbalance_ratio = imbalance_ratio

        # Load CIFAR-10 dataset
        train_data = CIFAR100(root=self.dataset_dir, train=True, download=True)
        test_data = CIFAR100(root=self.dataset_dir, train=False, download=True)

        # Rebalance the training data to introduce class imbalance
        test_data = self._apply_imbalance(test_data)

        # Convert CIFAR-10 data into a list of Datum objects
        train = self._create_data_source(train_data)
        test = self._create_data_source(test_data)

        # Initialize the parent class
        super().__init__( test=test)

        # Set class names
        self._classnames = train_data.classes
        self.template = template



    def _apply_imbalance(self, data):
        """使用 N_max 和 N_min 计算失衡比率并生成类别失衡的数据集。"""
        # 计算每个类别的初始样本数量
        class_counts = {i: 0 for i in range(100)}
        for _, label in data:
            class_counts[label] += 1
        print(class_counts)
        # 计算最多样本类别的数量 N_max
        N_max = max(class_counts.values())

        # 根据给定的 imbalance_ratio 计算最少样本类别的数量 N_min
        N_min = int(N_max / self.imbalance_ratio)

        # 使用指数衰减函数计算每个类别的新样本数量
        new_class_counts = {
            i: int(N_max * (self.imbalance_ratio ** (-i / (len(class_counts) - 1))))
            for i in range(100)
        }

        # 为每个类别选择保留的样本索引
        indices_to_keep = []
        class_sample_counts = {i: 0 for i in range(100)}

        for index, (_, label) in enumerate(data):
            if class_sample_counts[label] < new_class_counts[label]:
                indices_to_keep.append(index)
                class_sample_counts[label] += 1

        # 创建一个新的数据集，仅保留选定的样本
        data.data = data.data[indices_to_keep]
        data.targets = np.array(data.targets)[indices_to_keep].tolist()
        print(class_sample_counts)
        return data

    def _create_data_source(self, cifar_data):
        """Create a list of Datum objects from CIFAR-10 data."""
        data_source = []
        for img, label in cifar_data:
            # Use a placeholder for impath since CIFAR-10 images are in memory
            impath = 'in_memory_image'
            classname = cifar_data.classes[label]
            domain = 0  # CIFAR-10 does not have multiple domains
            datum = Datum(impath=impath, label=label, domain=domain, classname=classname)
            # Attach the image directly to the Datum object
            datum.image = img
            data_source.append(datum)
        return data_source
