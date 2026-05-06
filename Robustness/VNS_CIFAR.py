import torch
import random
import torch.nn as nn
import time
import csv

from Networks import AF_Tools
from Networks.GetData import *
import os.path
import torch.optim as optim

from Networks.AlexNet_CIFAR import *
from torchattacks import FGSM, PGD
import numpy as np

Activation_Functions = AF_Tools.Activation_Functions_
operators = AF_Tools.operators_
operators_p = AF_Tools.operators_p_

EPOCHS = 20
LAYERS = 7
destination_path = '../save/cifar10/cifar10'
result_file_name = 'VNS_CIFAR_FGSM.csv'
trace = False

def train_model(model, data_loaders, dataset_sizes, criterion, optimizer, device):
    model = model.to(device)
    model.train()  # set train mode

    for epoch in range(EPOCHS):
        epoch_time = time.time()
        running_loss, running_corrects, total_checked = 0.0, 0, 0
        error_found = False
        
        for indx, (inputs, labels) in enumerate(data_loaders['train']):
            try:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_loss_total = loss.item() * inputs.size(0)
                running_loss += batch_loss_total
                running_corrects += torch.sum(preds == labels.data)
                total_checked += inputs.size(0)

            except Exception as e:
                error_found = True
                print('Error occurred during training batch:', str(e))
                break

        if error_found:
            return -1
        
        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() * 100 / total_checked
        
        print('Epoch {}/{} - Train Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, EPOCHS, epoch_loss, epoch_acc))
        print('Time taken for epoch:', time.time() - epoch_time)

    return 0

def test_model(model, data_loaders, criterion, device, dataset_sizes):
    model.eval()
    running_loss, running_corrects, total_checked = 0.0, 0, 0

    for indx, (inputs, labels) in enumerate(data_loaders['test']):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_checked += inputs.size(0)

    test_loss = running_loss / dataset_sizes['test']
    test_acc = running_corrects.double() * 100 / total_checked

    return test_loss, test_acc.item()

def get_attacks():
    return {'FGSM': FGSM, 'PGD': PGD}

def atk_by_name(atkname):
    attaks = get_attacks()
    return attaks[atkname]

class VNS:
    def __init__(self, data_loaders, dataset_sizes, atk_):
        self.dataloaders = data_loaders
        self.dataset_sizes = dataset_sizes
        self.atk, self.eps = atk_
        self.p1 = range(0, LAYERS)
        self.p2 = range(LAYERS, LAYERS * 2)
        self.p3 = range(LAYERS * 2, LAYERS * 3)
        self.best_state = np.random.choice(len(Activation_Functions), LAYERS * 3, replace=True)
        self.best_energy = float('inf')

    def local_search(self, state):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AlexNet_CIFAR(num_classes=10, states_=state).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        fname = destination_path + '_'.join(map(str, state)) + '.pth'

        if os.path.isfile(fname):
            model.load_state_dict(torch.load(fname))
        else:
            res = train_model(model, self.dataloaders, self.dataset_sizes, criterion, optimizer, device)
            if res < 0:
                return 100
            torch.save(model.state_dict(), fname)

        test_loss, test_acc = test_model(model, self.dataloaders, criterion, device, self.dataset_sizes)

        correct = 0
        total = 0
        model.eval()
        attack = atk_by_name(self.atk)
        a_t_k = attack(model, eps=self.eps)

        for indx, (images, labels) in enumerate(self.dataloaders['test']):
            perturb_images = a_t_k(images, labels).to(device)
            outputs = model(perturb_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum()

        rob_acc = 100 * float(correct) / total
        energy = 100 * (1 - rob_acc)

        with open(result_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([state, self.atk, self.eps, rob_acc, test_acc])

        print(state, self.atk, self.eps, test_acc, rob_acc)
        return energy

    def shake(self, state, k):
        new_state = state.copy()
        for _ in range(k):
            loc = random.randint(0, len(state) - 1)
            if loc in self.p1:
                new_state[loc] = random.choices(operators_p, weights=operators_p)[0]
                if new_state[loc] < 2:
                    new_state[loc + LAYERS * 2] = 0
            elif loc in self.p2:
                new_state[loc] = random.randint(0, len(Activation_Functions) - 1)
            elif loc in self.p3:
                new_state[loc] = random.randint(0, len(Activation_Functions) - 1)
                if new_state[loc - LAYERS * 2] < 2:
                    new_state[loc - LAYERS * 2] = random.randint(2, 4)
        return new_state

    def VNS(self):
        k_max = 3
        while True:
            k = 1
            while k <= k_max:
                new_state = self.shake(self.best_state, k)
                new_energy = self.local_search(new_state)
                if new_energy < self.best_energy:
                    self.best_state = new_state
                    self.best_energy = new_energy
                    k = 1
                else:
                    k += 1
            print('Current best energy:', self.best_energy)

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    data_loaders_, dataset_sizes_ = get_cifar10_data_32(512, 256)
    atk_eps_s = [("FGSM", 1 / 255)]
    
    with open(result_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['State', 'Attack', 'Epsilon', 'Robust Accuracy', 'Test Accuracy'])

    for atk_eps_ in atk_eps_s:
        vns = VNS(data_loaders_, dataset_sizes_, atk_eps_)
        vns.VNS()
