import torch
import torch.nn as nn
import load_data as ld
import time
import argparse

dataloaders, dataset_sizes = ld.load_dataset()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Evaluation(object):
    def __init__(self, args):
        super().__init__()

        self.crit = nn.CrossEntropyLoss()
        self.model = torch.load(args.model_pth)
        self.model.eval()

    def eval_model(self):
        start_time = time.time()

        class_names = ['0', '1', '2', '3', '4']
        milk_list = {'0': '딸기우유', '1': '바나나우유', '2': '초코우유', '3': '커피우유', '4': '흰우유'}

        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0

            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.crit(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 한 배치의 첫 번째 이미지에 대하여 결과 시각화
                print(f'[예측 결과: {milk_list[class_names[preds[0]]]}] (실제 정답: {milk_list[class_names[labels.data[0]]]})')

            epoch_loss = torch.div(running_loss, dataset_sizes['test'])
            epoch_acc = torch.div(running_corrects, dataset_sizes['test']) * 100.

            print('[Test Phase] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch_loss, epoch_acc,
                                                                                time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters for model')

    parser.add_argument('--model_pth', type=str, default='milkathon_epoch100.pt',
                        help='path where the model exists')

    args = parser.parse_args()

    evaluation = Evaluation(args)
    evaluation.eval_model()