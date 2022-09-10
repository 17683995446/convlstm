from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch


class MovingMNIST(data.Dataset):
    """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    urls = [
        'https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'moving_mnist_train.pt'
    test_file = 'moving_mnist_test.pt'

    def __init__(self, root, train=True, split=1000, transform=None, target_transform=None, download=True, seq_len=10,
                 horizon=10):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train  # training set or test set
        self.seq_len = seq_len
        self.horizon = horizon
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def crop_image(self, img):
        """ img: [T, C, H, W]"""

        if self.crop_size is not None:
            return img[..., :self.crop_size, :self.crop_size]
        else:
            return img

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """

        # need to iterate over time
        def _transform_time(data):
            new_data = None
            for i in range(data.size(0)):
                img = Image.fromarray(data[i].numpy(), mode='L')
                new_data = self.transform(img) if new_data is None else torch.cat([self.transform(img), new_data],
                                                                                  dim=0)
            return new_data

        if self.train:
            seq = self.train_data[index, :self.seq_len]
            target = self.train_data[index, self.seq_len: (self.seq_len + self.horizon)]
        else:
            seq = self.test_data[index, :self.seq_len]
            target = self.test_data[index, self.seq_len:(self.seq_len + self.horizon)]
        if self.transform is not None:
            seq = _transform_time(seq)
        if self.target_transform is not None:
            target = _transform_time(target)


        '''
        T,W,H = seq.shape
        P = 18
        # print(P)
        seq_3d = torch.zeros((T,P,W//2,H//2),dtype=torch.float)
        target_3d = torch.zeros((T,P,W//2,H//2),dtype=torch.float)

        # for t in range(T):
        #     for m in range(0,64,2):
        #         for n in range(1,64,2):
        #             seq_3d[t][0][m//2][n//2]=seq[t][m][n]
        #             target_3d[t][0][m//2][n//2]=target[t][m][n]
        #             seq_3d[t][10][m//2][n//2]=seq[t][m][n]
        #             target_3d[t][10][m//2][n//2]=target[t][m][n]
        #             seq_3d[t][20][m//2][n//2]=seq[t][m][n]
        #             target_3d[t][20][m//2][n//2]=target[t][m][n]
        
        
        for t in range(T):
            k=0
            for i in range(0,3,1):
                if i%2==0:
                    for j in range(0,3,1):
                        seq_3d[t][k] = seq[t][16*i:16*i+32,16*j:16*j+32]
                        target_3d[t][k] = target[t][16*i:16*i+32,16*j:16*j+32]
                        k+=1
                else:
                    for j in range(2,-1,-1):
                        seq_3d[t][k] = seq[t][16*i:16*i+32,16*j:16*j+32]
                        target_3d[t][k] = target[t][16*i:16*i+32,16*j:16*j+32]
                        k+=1
        
        for t in range(T):
            k = 9
            for j in range(2,-1,-1):
                if j%2==1:
                    for i in range(0,3,1):
                        seq_3d[t][k] = seq[t][16*i:16*i+32,16*j:16*j+32]
                        target_3d[t][k] = target[t][16*i:16*i+32,16*j:16*j+32]
                        k+=1
                else:
                    for i in range(2,-1,-1):
                        seq_3d[t][k] = seq[t][16*i:16*i+32,16*j:16*j+32]
                        target_3d[t][k] = target[t][16*i:16*i+32,16*j:16*j+32]
                        k+=1
        
        seq_3d = seq_3d.unsqueeze(1)  # adding channel dimension
        target_3d = target_3d.unsqueeze(1)
        
        seq_3d = torch.zeros((T,W,H,16),dtype=torch.float)
        target_3d = torch.zeros((T,W,H,16),dtype=torch.float)
        for i in range(T):
            for j in range(W):
                for k in range(H):
                    for m in range(seq[i][j][k]//16):
                        seq_3d[i][j][k][m]=16.0
                    for m in range(target[i][j][k]//16):
                        target_3d[i][j][k][m]=16.0
                    if seq[i][j][k]%16>0 :
                        seq_3d[i][j][k][int(seq[i][j][k]//16)]= seq[i][j][k]%16.0
                    if target[i][j][k]%16 >0 :
                        target_3d[i][j][k][int(seq[i][j][k]//16)]= target[i][j][k]%16.0
        # seq = seq.unsqueeze(1)  # adding channel dimension
        # target = target.unsqueeze(1)
        seq_3d = seq_3d.unsqueeze(1)  # adding channel dimension
        target_3d = target_3d.unsqueeze(1)
        
        # return seq / 255.0, target / 255.0
        
        return seq_3d / 1.0, target_3d / 1.0
        '''
        seq = seq.unsqueeze(1)  # adding channel dimension
        target = target.unsqueeze(1)

        return seq / 255.0, target / 255.0

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            print("filename",filename)
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[:-self.split]
        )
        test_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[-self.split:]
        )

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train_set = MovingMNIST(root='.data/mnist', train=True, download=True)
    x, y = train_set[0]
    print(x.shape, y.shape)
    # fig, axis = plt.subplots(2, 10)
    #
    # for j in range(10):
    #     axis[0][j].imshow(x[j])
    # for j in range(10):
    #     axis[1][j].imshow(y[j])
    # fig.show()
    # plt.show()
    # test_set = MovingMNIST(root='.data/mnist', train=False, download=True)
