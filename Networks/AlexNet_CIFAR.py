# Load in relevant libraries, and alias where appropriate

import torch.nn as nn

from .New_AF import new_af


class AlexNet_CIFAR(nn.Module):
    def __init__(self, num_classes, states_):
        super(AlexNet_CIFAR, self).__init__()
        lyrs = 7
        cur_lyr = 0

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            new_af(states_[cur_lyr], states_[cur_lyr + lyrs], states_[cur_lyr + lyrs * 2]),
            nn.MaxPool2d(2))
        cur_lyr += 1
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            new_af(states_[cur_lyr], states_[cur_lyr + lyrs], states_[cur_lyr + lyrs * 2]),
            nn.MaxPool2d(2))
        cur_lyr += 1
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            new_af(states_[cur_lyr], states_[cur_lyr + lyrs], states_[cur_lyr + lyrs * 2]))
        cur_lyr += 1
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            new_af(states_[cur_lyr], states_[cur_lyr + lyrs], states_[cur_lyr + lyrs * 2]))
        cur_lyr += 1
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            new_af(states_[cur_lyr], states_[cur_lyr + lyrs], states_[cur_lyr + lyrs * 2])
            #,            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        cur_lyr += 1
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*4*4, 4096),
            new_af(states_[cur_lyr], states_[cur_lyr + lyrs], states_[cur_lyr + lyrs * 2]))
        cur_lyr += 1
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            new_af(states_[cur_lyr], states_[cur_lyr + lyrs], states_[cur_lyr + lyrs * 2]))
        cur_lyr += 1
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
