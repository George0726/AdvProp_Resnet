import torch
import net_rectified
import os
# from resnet18 import test
from attacker import PGDAttacker,NoOpAttacker
from wrn import wrn_50_2
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
from train import MixBatchNorm2d



normal_dir = "./checkpoint"
aux_dir = "./aux_bn"

model_bn = net_rectified.__dict__['wide_resnet50_2'](num_classes=10 ,norm_layer=MixBatchNorm2d)
model_bn.set_attacker(PGDAttacker)
model_bn.set_mixbn(False)
# filepath_bn = os.path.join(normal_dir,'model_best.pth.tar')
# model_bn.load_state_dict(torch.load(filepath_bn))

# model_aux = net.__dict__['resnet34'](num_classes=10 ,norm_layer=None)
# model_aux.set_attacker(NoOpAttacker)
# model_aux.set_mixbn(False)
# filepath_aux = os.path.join(aux_dir,'aux_checkpoint.pth.tar')
# model_aux.load_state_dict(torch.load(filepath_aux))

model_bn_cifar = wrn_50_2(num_class = 10)
print(len(model_bn_cifar.state_dict().keys()))
print(len(model_bn_cifar.state_dict().keys()))
# model_aux_cifar = ResNet34_cifar(num_classes = 10)
# for i in range(len(model_bn_cifar.state_dict().keys())):
#     print(list(model_bn_cifar.state_dict().keys())[i])
#     print(list(model_bn.state_dict().keys())[i])

# for key in model_bn_cifar.state_dict().keys():

    # if key in model_bn.state_dict().keys():
    #     model_bn_cifar.state_dict()[key].copy_(model_bn.state_dict()[key])

    # elif "shortcut" in key:
    #     key_modified = key.replace('shortcut','downsample')
    #     model_bn_cifar.state_dict()[key].copy_(model_bn.state_dict()[key_modified])

    # elif "linear" in key:
    #     key_modified = key.replace('linear', 'fc')
    #     model_bn_cifar.state_dict()[key].copy_(model_bn.state_dict()[key_modified])

    # else:
    #     print("WRONG KEY! ", key)

# for key in model_aux_cifar.state_dict().keys():

#     if key in model_aux.state_dict().keys():
#         model_aux_cifar.state_dict()[key].copy_(model_aux.state_dict()[key])

#     elif "shortcut" in key:
#         key_modified = key.replace('shortcut', 'downsample')
#         model_aux_cifar.state_dict()[key].copy_(model_aux.state_dict()[key_modified])

#     elif "linear" in key:
#         key_modified = key.replace('linear', 'fc')
#         model_aux_cifar.state_dict()[key].copy_(model_aux.state_dict()[key_modified])

#     else:
#         print("WRONG KEY! ", key)


#save the model
# filepath_bn = os.path.join('DI_bn_checkpoint.pth.tar')
# torch.save(model_bn_cifar.state_dict(), filepath_bn)

# filepath_aux = os.path.join('DI_aux_checkpoint.pth.tar')
# torch.save(model_aux_cifar.state_dict(), filepath_aux)

# test for loading
# model_test_bn = ResNet34_cifar(num_classes = 10)
# model_test_bn.load_state_dict(torch.load(filepath_bn))

# model_test_aux = ResNet34_cifar(num_classes = 10)
# model_test_aux.load_state_dict(torch.load(filepath_aux))

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# val_dataset = datasets.CIFAR10(root="./data", train=False, download=True,
#                                transform=transform_test)  # change to training transform
# val_loader = torch.utils.data.DataLoader(
#     val_dataset,
#     batch_size=64, shuffle=True,
#     num_workers=0, pin_memory=True)
# criterion = nn.CrossEntropyLoss(reduction='none').cuda()

# model_bn.to('cuda')
# test_loss_bn, test_acc_bn = test(val_loader, model_bn, criterion, 0,True)

# print('BN Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss_bn, test_acc_bn))


#     # measure data loading time
# model_test_bn.to('cuda')
# # measure accuracy and record loss
# for batch_idx, (inputs, targets) in enumerate(val_loader):
#     if batch_idx> 1:
#         break

#     inputs, targets = inputs.to('cuda'), targets.to('cuda')
#     outputs = model_test_bn(inputs)
#     _, predicted = outputs.max(1)
#     print(targets,predicted)
#     #
#     # test_loss += loss.item()
#     # _, predicted = outputs.max(1)
#     # total += targets.size(0)
#     # correct += predicted.eq(targets).sum().item()