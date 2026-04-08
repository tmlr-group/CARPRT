import os

from .oxford_pets import OxfordPets
from .utils import DatasetBase
from .template import template

class StanfordCars(DatasetBase):

    dataset_dir = 'stanford_cars'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_StanfordCars.json')

        self.template = template

        test = OxfordPets.read_split(self.split_path, self.dataset_dir)

        super().__init__(test=test)