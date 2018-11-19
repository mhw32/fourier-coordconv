DATA_DIR = '/mnt/fs5/wumike/datasets'
DATA_OPTIONS = ['DynamicMNIST', 'PerturbMNIST', 'FashionMNIST',
                'Histopathology', 'CelebA', 'SVHN', 'CIFAR10']
CONV_OPTIONS = ['vanilla', 'coord']
DIST_OPTIONS = ['bernoulli', 'gaussian']
LABEL_OPTIONS = ['bernoulli', 'categorical']
DATA_SHAPE = {
    'DynamicMNIST': (1, 28, 28),
    'PerturbMNIST': (1, 32, 32),
    'FashionMNIST': (1, 28, 28),
    'Histopathology': (1, 28, 28),
    'CelebA': (3, 64, 64),
    'SVHN': (3, 32, 32),
    'CIFAR10': (3, 32, 32),
}
DATA_DIST = {
    'DynamicMNIST': 'bernoulli', 
    'PerturbMNIST': 'bernoulli', 
    'FashionMNIST': 'bernoulli',            
    'Histopathology': 'bernoulli', 
    'CelebA': 'bernoulli', 
    'SVHN': 'bernoulli', 
    'CIFAR10': 'bernoulli',
}