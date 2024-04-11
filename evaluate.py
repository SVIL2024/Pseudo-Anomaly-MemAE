import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import Reconstruction3DDataLoader
from utils import *
import glob
import time
from memAE import *
import argparse


parser = argparse.ArgumentParser(description="VAD")
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--num_workers_test', type=int, default=0, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model')
parser.add_argument('--exp_dir', type=str, default='log', help='basename of folder to save result')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate phase 1')
parser.add_argument('--img_dir', type=str, default=None, help='save image file')

parser.add_argument('--mem_usage', default=[False, False, False, True], type=str)
parser.add_argument('--skip_ops', default=["none", "concat", "concat"], type=str)

parser.add_argument('--print_score', action='store_true', help='print score')
parser.add_argument('--vid_dir', type=str, default=None, help='save video frames file')
parser.add_argument('--print_time', action='store_true', help='print forward time')

args = parser.parse_args()

exp_dir = args.exp_dir
exp_dir += 'lr' + str(args.lr) if args.lr != 1e-4 else ''
exp_dir += 'weight'
exp_dir += '_recon'



loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
model = ML_MemAE_SC(num_in_ch=1, features_root=32,
                 mem_dim=2000, shrink_thres=0.0005,
                 mem_usage=args.mem_usage, skip_ops=args.skip_ops)


model_dict = torch.load(os.path.join('exp_AE', args.dataset_type, 'logweight_recon', 'model_00.pth'))
try:
    model_weight = model_dict['model']
    model.load_state_dict(model_weight.state_dict())
except KeyError:
    model.load_state_dict(model_dict['model_statedict'])

model.cuda()
labels = np.load('./frame_labels_'+args.dataset_type+'.npy', allow_pickle=True)


# Loading dataset
test_folder = os.path.join(args.dataset_path, args.dataset_type, 'testing')

img_extension = '.tif' if args.dataset_type == 'ped1' else '.jpg'
test_dataset = Reconstruction3DDataLoader(test_folder, transforms.Compose([transforms.ToTensor(),]),
                                          resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, img_extension=img_extension)
test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*/')))

for video in videos_list:
    video_name = video.split('\\')[-2]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + img_extension))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}



# Setting for video anomaly detection

for video in sorted(videos_list):
    video_name = video.split('\\')[-2]
    labels_list = np.append(labels_list, labels[0][8+label_length:videos[video_name]['length']+label_length-7])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []

label_length = 0
video_num = 0


label_length += videos[videos_list[video_num].split('\\')[-2]]['length']

model.eval()



for k,(imgs) in enumerate(test_batch):

    if k == label_length-15*(video_num+1):
        video_num += 1
        label_length += videos[videos_list[video_num].split('\\')[-2]]['length']

    imgs = Variable(imgs).cuda()
    with torch.no_grad():
        outs1 = model(imgs)
        outs = outs1['recon']
        loss_mse = loss_func_mse(outs[0, :, 8], imgs[0, :, 8])

    loss_pixel = torch.mean(loss_mse)
    mse_imgs = loss_pixel.item()

    psnr_list[videos_list[video_num].split('\\')[-2]].append(psnr(mse_imgs))


# Measuring the abnormality score (S) and the AUC
anomaly_score_total_list = []
vid_idx = []
for vi, video in enumerate(sorted(videos_list)):
    video_name = video.split('\\')[-2]
    score = anomaly_score_list(psnr_list[video_name])
    anomaly_score_total_list += score
    vid_idx += [vi for _ in range(len(score))]

anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

print('vididx,frame,anomaly_score,anomaly_label')
for a in range(len(anomaly_score_total_list)):
    print(str(vid_idx[a]), ',', str(a), ',', 1-anomaly_score_total_list[a], ',', labels_list[a])



print('The result of ', args.dataset_type)
print('model_00th_AUC: ', accuracy*100, '%')
print('----------------------------------------')