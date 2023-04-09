### Imports ###

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.io import loadmat
import os


### Classes ###

class MPIIFaceGaze(Dataset):
    def __init__(self, img_transform, mode='train', data_src='../data/MPIIFaceGaze'):
        if mode not in ['train', 'test', 'val']:
            raise ValueError('Invalid Split %s' % mode)
        self.mode = mode
        self.data_src = data_src
        self.img_transform = img_transform

        self.images = []
        self.labels = {}
        for p in os.listdir(self.data_src):
            person_path = os.path.join(self.data_src, p)
            if not os.path.isdir(person_path):
                continue

            for d in os.listdir(person_path):
                if d[:3] == 'day':
                    day_path = os.path.join(person_path, d)
                    for img in os.listdir(day_path):
                        self.images.append(os.path.join(day_path, img))

            f = np.loadtxt(os.path.join(person_path, p+'.txt'), delimiter=' ', dtype=str)
            for d in f:
                self.labels[os.path.join(p, *d[0].split('/'))] = np.concatenate((d[3:15], d[-4:-1])).astype(np.float32)
        # x = 'p11\\day12\\0158.jpg'
        # mat = loadmat(os.path.join(self.data_src, 'p11/Calibration/monitorPose.mat'))
        # print(mat)
        # mat = loadmat(os.path.join(self.data_src, 'p11/Calibration/screenSize.mat'))
        # print(mat)
        # print(self.labels[x])
        # plt.imshow(Image.open(os.path.join(self.data_src, x)).convert('RGB'))
        # plt.show()

        # dists = (np.array(list(self.labels.values()))**2).sum(axis=1)**0.5
        # names = np.array(list(self.labels.keys()))[np.argsort(dists)]
        # plt.hist(dists, bins=500)
        # plt.xlim(0, 400)
        # plt.show()
        # for n in names[:10]:
        #     plt.imshow(Image.open(os.path.join(self.data_src, n)).convert('RGB'))
        #     plt.show()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')

        # x = np.array(img).sum(axis=2)
        # x[x < 60] = 0
        # x = np.nonzero(x)
        # img = img.crop((np.min(x[1]), np.min(x[0]), np.max(x[1]), np.max(x[0])))

        data = self.labels[self.images[index][len(self.data_src)+1:]]

        return img, data


### Run Code ###

if __name__ == "__main__":
    dataset = MPIIFaceGaze(None)
    print(len(dataset))
    print(list(dataset.labels.keys())[0])
    for i in np.random.choice(range(len(dataset)), 50):
        face, direct = dataset[i]
        print(np.array(face).shape)
        print(direct)
        plt.imshow(face)
        plt.scatter(*np.reshape(direct[:-3], (-1, 2)).T, 5, 'r')
        plt.show()
