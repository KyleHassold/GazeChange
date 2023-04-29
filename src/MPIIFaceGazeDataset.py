### Imports ###
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os


### Consts ###

data_points = {  # Source: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation#c12823
    'gaze_screen_pos': (0, 2),  # Gaze location on the screen coordinate in pixels
    'eye_pos': (2, 10),         # (x,y) position for the four eye corners
    'mouth_pos': (10, 14),      # (x,y) position for the two mouth corners
    'head_pos': (14, 20),       # The estimated 3D head pose in the camera coordinate system based on 6 points-based
                                # 3D face model, rotation and translation
    'face_center': (20, 23),    # Face center in the camera coordinate system
    'gaze_target': (23, 26)     # The 3D gaze target location in the camera coordinate system
}

base_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((300, 300), antialias=True),])
                                      # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


### Classes ###

class MPIIFaceGaze(Dataset):
    def __init__(self, img_transform=base_transforms, data_src='../data/MPIIFaceGaze', crop=True,
                 needed_labels=('eye_pos',), verbose=False):
        self.img_transform = img_transform
        self.needed_labels = needed_labels
        self.data_src = data_src
        self.crop = crop

        self.images = []
        self.labels = []
        self.person_indexes = [0]
        for p in sorted(os.listdir(self.data_src)):
            # Confirm current item is a folder
            person_path = os.path.join(self.data_src, p)
            if not os.path.isdir(person_path):
                continue
            if verbose:
                print(f"Loading images of Person {p}:")

            # Store the labels for the current person
            labels_dict = {}
            f = np.loadtxt(os.path.join(person_path, p + '.txt'), delimiter=' ', dtype=str)
            for d in f:
                labels_dict[os.path.join(p, *d[0].split('/'))] = d[1:-1].astype(np.float32)

            # Loop through each day
            for d in os.listdir(person_path):
                if d[:3] == 'day':
                    day_path = os.path.join(person_path, d)
                    # Gather image paths and associated labels for each image
                    for img in os.listdir(day_path):
                        self.images.append(os.path.join(day_path, img))
                        self.labels.append(labels_dict[os.path.join(p, d, img)])
            self.person_indexes.append(len(self.images))
        if verbose:
            print(f"Loaded {len(self.images)} total images")
            print("Person changes at " + str(self.person_indexes))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        data = np.array(self.labels[index])

        if self.crop:
            # Find the bounds of the faces
            x = np.array(img).sum(axis=2)
            x[x < 60] = 0  # Image boundaries are imprecise
            x = np.nonzero(x)
            img = img.crop((np.min(x[1]), np.min(x[0]), np.max(x[1]), np.max(x[0])))
            # Adjust any labels corresponding to pixel values
            data[[2, 4, 6, 8, 10, 12]] -= np.min(x[1])
            data[[3, 5, 7, 9, 11, 13]] -= np.min(x[0])

        # Apply image transforms
        if self.img_transform is not None:
            img_size = img.size
            img = self.img_transform(img)

            # Correct data (assuming only resize)
            if img.size()[1] != img_size[0]:
                data[[2, 4, 6, 8, 10, 12]] *= img.size()[1]/img_size[0]
            if img.size()[2] != img_size[1]:
                data[[3, 5, 7, 9, 11, 13]] *= img.size()[2]/img_size[1]

        # Collect desired data
        data = np.reshape([data[data_points[p][0]:data_points[p][1]] for p in self.needed_labels], -1)

        return img, data


def get_dataloaders(batch_size, dataset=None, split=0.8, shuffle=0, person_val=()):
    # Initialize dataset if not given
    dataset = MPIIFaceGaze() if dataset is None else dataset

    # Collect indices based on if a given person should be included
    train_indices, val_indices = [], []
    for i in range(len(dataset.person_indexes)-1):
        new_indexes = list(range(dataset.person_indexes[i], dataset.person_indexes[i+1]))
        if i in person_val:
            val_indices += new_indexes
        else:
            train_indices += new_indexes

    # Shuffle train indices for different split
    if shuffle:
        np.random.seed(shuffle)
        np.random.shuffle(train_indices)

    # Calculate and perform split on remaining train data
    split = int(split * len(train_indices))
    val_indices += train_indices[split:]
    train_indices = train_indices[:split]

    # Create samplers and associated dataloaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, validation_loader


def crop_eyes(imgs, eye_locs):
    eye_imgs = []
    for face, eyes in zip(imgs, eye_locs.to(torch.int)):
        top = max(min(eyes[1::2]).item() - 20, 0)
        left = max(min(eyes[::2]).item() - 20, 0)
        eye_imgs.append(face[:, top:top + 48, left:left + 160])
    return torch.stack(eye_imgs)


def insert_eyes(imgs, eye_locs, eyes):
    eye_imgs = []
    for face, locs, eye in zip(imgs, eye_locs.to(torch.int), eyes):
        top = max(min(locs[1::2]).item() - 20, 0)
        left = max(min(locs[::2]).item() - 20, 0)
        # face[:, top:top + 48, left:left + 160] = eye
        new_img = torch.cat((face[:, top:top + 48, 0:left], eye, face[:, top:top + 48, left+160:]), dim=2)
        new_img = torch.cat((face[:, 0:top, :], new_img, face[:, top + 48:, :]), dim=1)
        eye_imgs.append(new_img)
    return torch.stack(eye_imgs)


### Run Code ###

if __name__ == "__main__":
    dataset = MPIIFaceGaze(verbose=True)
    print(dataset[0][1])

    train_data, val_data = get_dataloaders(4, dataset)
    for face, marks in train_data:
        marks = marks.to(torch.int)
        plt.imshow(face[0].permute(1, 2, 0))
        plt.scatter(*np.reshape(marks[0], (-1, 2)).T, 5, 'r')
        plt.show()
        eyes = crop_eyes(face, marks)
        plt.imshow(eyes[0].permute(1, 2, 0))
        plt.show()
