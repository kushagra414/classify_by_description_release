from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class PetsDataset(OxfordIIITPet):
    def __init__(self, root, transform=None, split="trainval"):
        super(PetsDataset, self).__init__(root, split=split, transform=transform)
        self.class_to_species = dict()

    def get_species(self, file_path):
        os.path.join(self.root, file_path)
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                class_id, _, species, _ = line.strip('\n').split(' ', 3)
                old_class_id = ' '.join(class_id.split('_')[0: -1])
                class_id = ' '.join(class_id.split('_')[0: -1]).title()
                if int(species) == 1:
                    self.class_to_species[class_id] = 'Cat'
                else:
                    self.class_to_species[class_id] = 'Dog'
        return self.class_to_species

def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        # transforms.Normalize((0.4827, 0.4472, 0.3974), (0.2289, 0.2260, 0.2275)),  # Oxford Pets
    ])
