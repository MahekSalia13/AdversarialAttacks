from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
import torch


def get_mnist_data227(train_batch_size_ = 60000, test_batch_size_ = 10000):

    transform_train = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.1307,), std=(0.3081,))])



    transform_test = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.1325,), std=(0.3105,))])

    # The standard output of the torchvision MNIST data set is [0,1] range, which
    # is what we want for later processing. All we need for a transform, is to
    # translate it to tensors.

    #transform_train=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))  ])
    # We first download the train and test datasets if necessary and then load them into pytorch dataloaders.
    mnist_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
    mnist_test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_test, download=True)

    dataset_sizes = {'train': mnist_train_dataset.__len__(), 'test': mnist_test_dataset.__len__()}  # a dictionary to keep both train and test datasets

    print(dataset_sizes)
    #    mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train_dataset, batch_size=32, shuffle=True)
    #    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test_dataset, batch_size=1, shuffle=True)
    mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train_dataset, batch_size=train_batch_size_, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test_dataset, batch_size=test_batch_size_, shuffle=True)

    dataloaders = {'train': mnist_train_loader, 'test': mnist_test_loader}  # a dictionary to keep both train and test loaders

    return dataloaders, dataset_sizes



def get_mnist_data(train_batch_size_ = 60000, test_batch_size_ = 10000):
    transform_train = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

    #transform_train = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(),
    #                                      transforms.Normalize(mean=(0.1307,), std=(0.3081,))])


    transform_test = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.1325,), std=(0.3105,))])

    #transform_test = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(),
    #                                     transforms.Normalize(mean=(0.1325,), std=(0.3105,))])

    # The standard output of the torchvision MNIST data set is [0,1] range, which
    # is what we want for later processing. All we need for a transform, is to
    # translate it to tensors.

    #transform_train=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))  ])
    # We first download the train and test datasets if necessary and then load them into pytorch dataloaders.
    mnist_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
    mnist_test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_test, download=True)

    dataset_sizes = {'train': mnist_train_dataset.__len__(), 'test': mnist_test_dataset.__len__()}  # a dictionary to keep both train and test datasets

    print(dataset_sizes)
    #    mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train_dataset, batch_size=32, shuffle=True)
    #    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test_dataset, batch_size=1, shuffle=True)
    mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train_dataset, batch_size=train_batch_size_, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test_dataset, batch_size=test_batch_size_, shuffle=True)

    dataloaders = {'train': mnist_train_loader, 'test': mnist_test_loader}  # a dictionary to keep both train and test loaders

    return dataloaders, dataset_sizes


def get_mnist_data_without_normalize(train_batch_size_ = 60000, test_batch_size_ = 10000):
    transform_train = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

    #transform_train = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(),
    #                                      transforms.Normalize(mean=(0.1307,), std=(0.3081,))])


    transform_test = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()])

    #transform_test = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(),
    #                                     transforms.Normalize(mean=(0.1325,), std=(0.3105,))])

    # The standard output of the torchvision MNIST data set is [0,1] range, which
    # is what we want for later processing. All we need for a transform, is to
    # translate it to tensors.

    #transform_train=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))  ])
    # We first download the train and test datasets if necessary and then load them into pytorch dataloaders.
    mnist_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
    mnist_test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_test, download=True)

    dataset_sizes = {'train': mnist_train_dataset.__len__(), 'test': mnist_test_dataset.__len__()}  # a dictionary to keep both train and test datasets

    print(dataset_sizes)
    #    mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train_dataset, batch_size=32, shuffle=True)
    #    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test_dataset, batch_size=1, shuffle=True)
    mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train_dataset, batch_size=train_batch_size_, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test_dataset, batch_size=test_batch_size_, shuffle=True)

    dataloaders = {'train': mnist_train_loader, 'test': mnist_test_loader}  # a dictionary to keep both train and test loaders

    return dataloaders, dataset_sizes


def get_imagenettiny_data(data, name, transform):
    from torchvision import transforms as T
    from torch.utils.data import DataLoader

    if data is None:
        return None

    # Read image files to pytorch dataset using ImageFolder, a generic data
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device

    if torch.cuda.is_available():
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}

    # Wrap image dataset (defined above) in dataloader
    dataloader = DataLoader(dataset, batch_size=512,
                            shuffle=(name == "train"),
                            **kwargs)

    return dataloader



def get_cifar10_data_32(train_batch_size_ = 512, test_batch_size_ = 256):

    # creating a dinstinct transform class for the train, validation and test dataset
    transform_train = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=int(32 / 8)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),]
        )


    transform_test = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    '''
    tranform_train = transforms.Compose(
        [transforms.Resize((227, 227)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor()])
    tranform_test = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()])
    '''
    # preparing the train, validation and test dataset
    torch.manual_seed(4230)
    train_ds = CIFAR10("data/", train=True, download=True, transform=transform_train)  # 40,000 original images + transforms
    test_ds = CIFAR10("data/", train=False, download=True, transform=transform_train)  # 10,000 images

    dataset_sizes = {'train': train_ds.__len__(), 'test': test_ds.__len__()}  # a dictionary to keep both train and test datasets

    # passing the train, val and test datasets to the dataloader
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size_, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size_, shuffle=False)

    dataloaders = {'train': train_dl, 'test': test_dl}

    return dataloaders, dataset_sizes




def get_cifar10_data_227(train_batch_size_ = 512, test_batch_size_ = 256):

    # creating a dinstinct transform class for the train, validation and test dataset
    transform_train = transforms.Compose(
        [transforms.Resize((227, 227)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    '''
    tranform_train = transforms.Compose(
        [transforms.Resize((227, 227)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor()])
    tranform_test = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()])
    '''
    # preparing the train, validation and test dataset
    torch.manual_seed(1243)
    train_ds = CIFAR10("data/", train=True, download=True, transform=transform_train)  # 40,000 original images + transforms
    test_ds = CIFAR10("data/", train=False, download=True, transform=transform_test)  # 10,000 images

    dataset_sizes = {'train': train_ds.__len__(), 'test': test_ds.__len__()}  # a dictionary to keep both train and test datasets

    # passing the train, val and test datasets to the dataloader
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size_, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size_, shuffle=False)

    dataloaders = {'train': train_dl, 'test': test_dl}

    return dataloaders, dataset_sizes




def get_cifar10_data_old_without_normalize(train_batch_size_ = 512, test_batch_size_ = 256, rndom_seed=0):

    # creating a dinstinct transform class for the train, validation and test dataset
    transform_train = transforms.Compose(
        [transforms.Resize((227, 227)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()])

    # preparing the train, validation and test dataset
    torch.manual_seed(rndom_seed)
    train_ds = CIFAR10("data/", train=True, download=True, transform=transform_train)  # 40,000 original images + transforms
    test_ds = CIFAR10("data/", train=False, download=True, transform=transform_test)  # 10,000 images

    dataset_sizes = {'train': train_ds.__len__(), 'test': test_ds.__len__()}  # a dictionary to keep both train and test datasets
    print(dataset_sizes)
    # passing the train, val and test datasets to the dataloader
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size_, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size_, shuffle=True)

    dataloaders = {'train': train_dl, 'test': test_dl}

    return dataloaders, dataset_sizes
