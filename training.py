import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import random
import argparse
from torchvision import models
import load_data as ld

# 나중에 device도 args에 넘겨주기
device = "cpu"

random_seed = 555
random.seed(random_seed)
torch.manual_seed(random_seed)

dataloaders, dataset_sizes = ld.load_dataset()


def transfer_learning():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 5)

    model_ft = model_ft.to(device)

    return model_ft


class Train(object):
    def __init__(self, args):
        super().__init__()

        self.model_pth = args.model_pth
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr

        self.model = transfer_learning()

        self.crit = nn.CrossEntropyLoss()
        self.optim = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.lr_scheduler = lr_scheduler.StepLR(self.optim, step_size=7, gamma=0.1)

    def train_model(self):

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.n_epochs):
            print('Epoch {}/{}'.format(epoch, self.n_epochs - 1))
            print('-' * 10)

            # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()  # 모델을 학습 모드로 설정
                else:
                    self.model.eval()  # 모델을 평가 모드로 설정

                running_loss = 0.0
                running_corrects = 0

                # 데이터를 반복
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 매개변수 경사도를 0으로 설정
                    self.optim.zero_grad()

                    # 순전파
                    # 학습 시에만 연산 기록을 추적
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.crit(outputs, labels)

                        # 학습 단계인 경우 역전파 + 최적화
                        if phase == 'train':
                            loss.backward()
                            self.optim.step()

                    # 통계
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    self.lr_scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # 모델을 깊은 복사(deep copy)함
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # 가장 나은 모델 가중치를 불러옴
        self.model.load_state_dict(best_model_wts)

        # 모델 저장
        torch.save(self.model, self.model_pth)

        return self.model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters for model')

    parser.add_argument('--n_epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('--model_pth', type=str, default='milkathon_epoch100.pt', help='path where the model exists')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    args = parser.parse_args()

    model = transfer_learning()
    model = Train(args)

    model.train_model()