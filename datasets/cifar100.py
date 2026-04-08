import os
from .utils import DatasetBase, Datum
from torchvision.datasets import CIFAR100
from .template import template


class CIFAR_100(DatasetBase):
    """CIFAR-10 dataset class as a subclass of DatasetBase."""

    dataset_dir = 'cifar-10'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # Load CIFAR-10 dataset
        train_data = CIFAR100(root=self.dataset_dir, train=True, download=True)
        test_data = CIFAR100(root=self.dataset_dir, train=False, download=True)

        # Convert CIFAR-10 data into a list of Datum objects
        train = self._create_data_source(train_data)
        test = self._create_data_source(test_data)

        # Initialize the parent class
        super().__init__(train_x=train, test=test)

        # Set class names
        self._classnames = train_data.classes
        self.template = template

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
