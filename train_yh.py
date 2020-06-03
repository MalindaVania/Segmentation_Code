import time
import os
import sublist
import ipdb
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

opt = TrainOptions().parse()



# Method = 'ImageOnly'
Method = opt.yh_data_model


'''raw_MRI_dir = '/home/mvania/Seg-Net/Datasets/MRI'
raw_MRI_seg_dir = '/home/mvania/Seg-Net/Datasets/Label'
raw_CT_dir = '/home/mvania/Seg-Net/Datasets/CT'
raw_CT_dir_test = '/home/mvania/Seg-Net/Datasets/CT_test'
sub_list_dir = './sublist'
'''

#B
raw_MRI_dir = '/home/mvania/Seg-Net/Dataset_MRISeg/Label'
#'/home/mvania/Seg-Net/Datasets_CTMRI/MRI' 
#'/home/mvania/Seg-Net/Datasets_CTSeg/Label'
raw_MRI_seg_dir = '/home/mvania/Seg-Net/Dataset_MRISeg/Label'
#'/home/mvania/Seg-Net/Datasets_CTMRI/Label' 
#'/home/mvania/Seg-Net/Datasets_CTSeg/Label'

#A
#raw_CT_dir = '/home/mvania/Seg-Net/Dataset_MRISeg/MRI'
#'/home/mvania/Seg-Net/Datasets_CTMRI/CT' 
#'/home/mvania/Seg-Net/Datasets_CTSeg/CT'
#raw_CT_dir_test = '/home/mvania/Seg-Net/Dataset_MRISeg/MRI'
#sub_list_dir = './sublist'
raw_CT_dir = '/home/mvania/Seg-Net/Dataset_CTSeg/Test2_case1'
raw_CT_dir_test = '/home/mvania/Seg-Net/Datasets_CTSeg/Test2_case1'
sub_list_dir = './sublist'


TrainOrTest = opt.yh_run_model #'Train' #


#evaluation
if TrainOrTest == 'Test':

    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.isTrain = False
    opt.phase = 'test'
    opt.no_dropout = True

    
    cycle_output_dir = opt.test_seg_output_dir
    opt.test_CT_dir = raw_CT_dir_test

    mkdir(sub_list_dir)
    sub_list_MRI = os.path.join(sub_list_dir, 'sublist_Phantom.txt')
    sub_list_CT = os.path.join(sub_list_dir, 'sublist_Phantom.txt')

    imglist_MRI = sublist.dir2list(raw_MRI_dir, sub_list_MRI)
    imglist_CT = sublist.dir2list(raw_CT_dir_test, sub_list_CT)

    #imglist_MRI, imglist_CT = sublist.equal_length_two_list(imglist_MRI, imglist_CT);

    # input the opt that we want
    opt.raw_MRI_dir = raw_MRI_dir
    #opt.raw_MRI_seg_dir = raw_MRI_seg_dir
    opt.raw_CT_dir = raw_CT_dir
    opt.imglist_MRI = imglist_MRI
    opt.imglist_CT = imglist_CT

    data_loader = CreateDataLoader(opt)
    #dataset = data_loader.dataset()
    dataset = data_loader.load_data(opt)
    dataset_size = len(data_loader)
    dataset_size = len(data_loader)


    print('#testing images = %d' % dataset_size)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images_to_dir(cycle_output_dir, visuals, img_path)