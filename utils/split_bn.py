import torch
import net_rectified
import os
from train import MixBatchNorm2d,test
from attacker import PGDAttacker,NoOpAttacker
from utils import mkdir_p
from collections import OrderedDict
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn

normal_dir = "./wrn_50_normal_bn"
# aux_dir = "./checkpoint_mixbn_cifar10"

#model pre-load
model0 = net_rectified.__dict__['wide_resnet50_2'](num_classes=10 ,norm_layer=MixBatchNorm2d)
model0.set_attacker(PGDAttacker)
model0.set_mixbn(True)
model0 = torch.nn.DataParallel(model0).cuda()

print('==> Resuming from checkpoint..')

checkpoint = torch.load("./checkpoint/model_best.pth.tar")
best_acc = checkpoint['best_acc']
start_epoch = checkpoint['epoch']
model0.load_state_dict(checkpoint['state_dict'])

#initialize a model without data parallel, remove the "modules" in keys
new_state_dict = OrderedDict()
for k, v in model0.state_dict().items():
    print(k)
    name = k[7:] # remove module.
    new_state_dict[name] = v

# construct and initialize the model
model = net_rectified.__dict__['wide_resnet50_2'](num_classes=10 ,norm_layer=MixBatchNorm2d)
model.set_attacker(PGDAttacker)
model.set_mixbn(True)
model.load_state_dict(new_state_dict)

model_bn = net_rectified.__dict__['wide_resnet50_2'](num_classes=10 ,norm_layer=None)
model_bn.set_attacker(NoOpAttacker)
model_bn.set_mixbn(False)
# model1 = torch.nn.DataParallel(model1).cuda()

model_aux = net_rectified.__dict__['wide_resnet50_2'](num_classes=10 ,norm_layer=None)
model_aux.set_attacker(NoOpAttacker)
model_aux.set_mixbn(False)
# model2 = torch.nn.DataParallel(model2).cuda()



for key in model.state_dict().keys():

    if 'bn' not in key:
        if key not in model_aux.state_dict():
            print(key)
        model_bn.state_dict()[key].copy_(model.state_dict()[key])
        model_aux.state_dict()[key].copy_(model.state_dict()[key])

    elif 'aux' not in key:
        model_bn.state_dict()[key].copy_(model.state_dict()[key])

    else:
        key_modified = key.replace(".aux_bn",'')
        model_aux.state_dict()[key_modified].copy_(model.state_dict()[key])

# save the model

print('==> save aux and normal model..')

if not os.path.isdir(normal_dir):
    mkdir_p(normal_dir)

# if not os.path.isdir(aux_dir):
#     mkdir_p(aux_dir)

#
filepath_bn = os.path.join(normal_dir,'bn_checkpoint.pth.tar')
# filepath_aux = os.path.join(aux_dir,'aux_checkpoint.pth.tar')
#
#
torch.save(model_bn.state_dict(), filepath_bn)
# torch.save(model_aux.state_dict(),filepath_aux)

# test for the model
# model_bn.to('cuda')
# model_aux.to('cuda')
# model.to('cuda')
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# val_dataset = datasets.CIFAR10(root="./data", train=False, download=True,
#                                transform=transform_test)  # change to training transform
# val_loader = torch.utils.data.DataLoader(
#     val_dataset,
#     batch_size=64, shuffle=True,
#     num_workers=0, pin_memory=True)
# criterion = nn.CrossEntropyLoss(reduction='none').cuda()
#
# test_loss_bn, test_acc_bn = test(val_loader, model_bn, criterion, 0,True)
# print('BN Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss_bn, test_acc_bn))
#
# test_loss_aux, test_acc_aux = test(val_loader, model_aux, criterion, 0,True)
# print('AUX Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss_aux, test_acc_aux))
#
# test_loss, test_acc = test(val_loader, model, criterion, 0,True)
# print('Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))










