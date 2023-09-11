import os
import shutil
import numpy as np

import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as T
import torchvision.datasets as datasets

import argparse
import re

import loader
import loader_bacs

from dlutils.sampler import MultipleSampler
from util.helpers import makedir
import push, model, train_and_test as tnt
from util import save
from util.log import create_logger
from util.preprocess import mean, std, preprocess_input_function

import settings_CUB
import settings_emma

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid',type=str, default='0,1,2,3')
parser.add_argument('-arch',type=str, default='vgg19')

parser.add_argument('-dataset',type=str,default="CUB")
#parser.add_argument('-times',type=str,default="test",help="experiment_run")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
print(os.environ['CUDA_VISIBLE_DEVICES'])

print('GPUs : ', torch.cuda.device_count())
#setting parameter
#experiment_run = args.times
experiment_run = settings_CUB.experiment_run
base_architecture = args.arch
dataset_name = args.dataset

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
#model save dir
model_dir = './tesnet_output/' + dataset_name+'/' + base_architecture + '/' + experiment_run + '/'

if os.path.exists(model_dir) is True:
    shutil.rmtree(model_dir)
makedir(model_dir)


shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings_CUB.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'models', base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'


# load the hyper param
if dataset_name == "CUB":
    #model param
    image_dir = settings_CUB.data_path
    num_classes = settings_CUB.num_classes
    img_size = int(settings_CUB.img_size)
    add_on_layers_type = settings_CUB.add_on_layers_type
    prototype_shape = settings_CUB.prototype_shape
    prototype_activation_function = settings_CUB.prototype_activation_function
    #datasets
    train_dir = settings_CUB.train_dir
    test_dir = settings_CUB.test_dir
    train_push_dir = settings_CUB.train_push_dir
    train_batch_size = settings_CUB.train_batch_size
    test_batch_size = settings_CUB.test_batch_size
    train_push_batch_size = settings_CUB.train_push_batch_size
    #optimzer
    joint_optimizer_lrs = settings_CUB.joint_optimizer_lrs
    joint_lr_step_size = settings_CUB.joint_lr_step_size
    warm_optimizer_lrs = settings_CUB.warm_optimizer_lrs

    last_layer_optimizer_lr = settings_CUB.last_layer_optimizer_lr
    # weighting of different training losses
    coefs = settings_CUB.coefs
    # number of training epochs, number of warm epochs, push start epoch, push epochs
    num_train_epochs = settings_CUB.num_train_epochs
    num_warm_epochs = settings_CUB.num_warm_epochs
    push_start = settings_CUB.push_start
    push_epochs = settings_CUB.pushchs
    np.random_seed(17)
    normalize = T.Normalize(mean=mean,std=std)
    resize = T.Resize(size=(img_size, img_size))
    h_flip = T.RandomHorizontalFlip(p=0.5)
    # rotation = T.RandomRotation(10)
    # affine = T.RandomAffine(degrees=15, shear=10)

    train_t = T.Compose([h_flip, resize, T.ToTensor(), normalize])
    # all datasets
    # train set
    train_dataset = loader.CroppedDataset(image_dir, sub_data='train', transform=train_t, push=False) #my loader
    print('dataset : ', len(train_dataset))
    # train_dataset = datasets.ImageFolder(
    #     train_dir,
    #     T.Compose([
    #         T.Resize(size=(img_size, img_size)),
    #         T.ToTensor(),
    #         normalize,
    #     ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, sampler=MultipleSampler(train_dataset, 10, 12), shuffle=False,
        num_workers=4, pin_memory=False)
    print('loader : ', len(train_loader))
    # push set
    train_push_dataset = loader.CroppedDataset(image_dir, sub_data='train', transform=T.Compose([resize, T.ToTensor()]), push=True)  #my loader
    # train_push_dataset = datasets.ImageFolder(
    #     train_push_dir,
    #     T.Compose([
    #         T.Resize(size=(img_size, img_size)),
    #         T.ToTensor(),
    #     ]))
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    # test set
    test_dataset = loader.CroppedDataset(image_dir, sub_data='valid', transform=T.Compose([resize, T.ToTensor(), normalize]), push=False)  #my loader
    # test_dataset = datasets.ImageFolder(
    #     test_dir,
    #     T.Compose([
    #         T.Resize(size=(img_size, img_size)),
    #         T.ToTensor(),
    #         normalize,
    #     ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

elif dataset_name == 'emma':
    mean_emma = (1229.04, 5304.82,  1708.66)
    std_emma = (206.16,  672.06, 1454.22)
    #model param
    print('emma')
    num_classes = settings_emma.num_classes
    img_size = settings_emma.img_size
    add_on_layers_type = settings_emma.add_on_layers_type
    prototype_shape = settings_emma.prototype_shape
    prototype_activation_function = settings_emma.prototype_activation_function
    #datasets
    image_dir = settings_emma.data_path
    # train_dir = settings_CUB.train_dir
    # test_dir = settings_CUB.test_dir
    # train_push_dir = settings_CUB.train_push_dir
    train_batch_size = settings_emma.train_batch_size
    test_batch_size = settings_emma.test_batch_size
    train_push_batch_size = settings_emma.train_push_batch_size
    #optimzer
    joint_optimizer_lrs = settings_emma.joint_optimizer_lrs
    joint_lr_step_size = settings_emma.joint_lr_step_size
    warm_optimizer_lrs = settings_emma.warm_optimizer_lrs

    last_layer_optimizer_lr = settings_emma.last_layer_optimizer_lr
    # weighting of different training losses
    coefs = settings_emma.coefs
    # number of training epochs, number of warm epochs, push start epoch, push epochs
    num_train_epochs = settings_emma.num_train_epochs
    num_warm_epochs = settings_emma.num_warm_epochs
    push_start = settings_emma.push_start
    push_epochs = settings_emma.push_epochs

    training_files, validation_files, test_files = loader_bacs.split_files('./datasets/EmmaOMDisruptors20230420_tif/')

    t_augment = T.Compose([T.RandomHorizontalFlip(p=0.5),T.RandomVerticalFlip(p=0.5), T.Normalize(mean=mean_emma,std=std_emma)])

    #load train data
    train_dataset = loader_bacs.EmmaDataset(training_files, transform=t_augment)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    #load train push data
    train_push_dataset = loader_bacs.EmmaDataset(training_files)
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    #load test data
    validation_data = loader_bacs.EmmaDataset(validation_files, transform=T.Compose([T.Normalize(mean=mean_emma, std=std_emma)]))
    test_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)


    


else:
    raise Exception("there are no settings file of datasets {}".format(dataset_name))

log(experiment_run)



# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))
log('savefolder: {0}'.format(experiment_run))

print('train batches: ', len(train_loader))
print('test batches: ', len(test_loader))

log("backbone architecture:{}".format(base_architecture))
log("basis concept size:{}".format(prototype_shape))
# construct the model
ppnet = model.construct_TesNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)


class_specific = True

# define optimizer
from settings_CUB import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings_CUB import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings_CUB import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

#best acc
best_acc = 0
best_epoch = 0
best_time = 0

# train the model
log('start training')
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))
    #stage 1: Embedding space learning
    #train
    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _,train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        _,train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)

    #test
    accu,test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)

    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.70, log=log)
    #stage2: Embedding space transparency
    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu,test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.70, log=log)
    #stage3: concept based classification
        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _,train_results= tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)

                accu,test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.70, log=log)
   
logclose()

