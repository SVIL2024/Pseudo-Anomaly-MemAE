import torchvision.transforms as transforms
from data import *
from utils import *
from memAE import *
import random
import argparse
import time


parser = argparse.ArgumentParser(description="VAD")
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--learning_rate_ped2', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--learning_rate_avenue', default=0.0000001, type=float, help='initial learning_rate')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the train loader')
parser.add_argument('--loss_m_weight', help='loss_m_weight', type=float, default=0.0002)
parser.add_argument('--dataset_type', type=str, default='ped2', choices=['ped2', 'avenue', 'shanghai'],
                    help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='basename of folder to save weights')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                    help='adam or sgd with momentum and cosine annealing lr')

parser.add_argument('--mem_usage', default=[False, False, False, True], type=str)
parser.add_argument('--skip_ops', default=["none", "concat", "none"], type=str)

parser.add_argument('--pseudo_anomaly_jump', type=float, default=0,
                    help='pseudo anomaly jump frame (skip frame) probability. 0 no pseudo anomaly')
parser.add_argument('--jump', nargs='+', type=int, default=[2], help='Jump for pseudo anomaly (hyperparameter s)')  # --jump 2 3


parser.add_argument('--model_dir', type=str, default=None, help='path of model for resume')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch. usually number in filename + 1')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

#Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')



args = parser.parse_args()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    device = torch.cuda.current_device()
channel_in = 1
if args.dataset_type == 'ped2':
    channel_in = 1
    learning_rate = args.learning_rate_ped2
    train_folder = os.path.join('UCSDped2', 'Train')

else:
    channel_in = 3
    learning_rate = args.learning_rate_avenue
    train_folder = os.path.join('Avenue', 'Train')
    args.epochs = args.epochs + 30

print(f'epochs:{args.epochs}')

exp_dir = args.exp_dir
# exp_dir += 'lr' + str(args.lr) if args.lr != 1e-4 else ''
exp_dir += 'weight'
exp_dir += '_recon'



print('exp_dir: ', exp_dir)

print(f'train_folder:{train_folder}')


# Loading dataset
img_extension = '.tif' if args.dataset_type == 'ped2' else '.jpg'
train_dataset = Reconstruction3DDataLoader(train_folder, transforms.Compose([transforms.ToTensor()]),
                                           resize_height=args.h, resize_width=args.w, dataset=args.dataset_type,
                                           img_extension=img_extension)
train_dataset_jump = Reconstruction3DDataLoaderJump(train_folder, transforms.Compose([transforms.ToTensor()]),
                                                resize_height=args.h, resize_width=args.w, dataset=args.dataset_type,
                                                jump=args.jump, img_extension=img_extension)

train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, drop_last=True)
train_batch_jump = data.DataLoader(train_dataset_jump, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers, drop_last=True)


print(f'len(train_batch):{len(train_batch)}')


# Report the training process
log_dir = os.path.join('./exp_AE', args.dataset_type, exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout

f = open(os.path.join(log_dir, 'log.txt'), 'a')
sys.stdout = f

torch.set_printoptions(profile="full")

loss_func_mse = nn.MSELoss(reduction='none')

loss_m_weight = args.loss_m_weight

if args.start_epoch < args.epochs:
    model = ML_MemAE_SC(num_in_ch=channel_in, features_root=32,
                        mem_dim=2000, shrink_thres=0.0005,
                        mem_usage=args.mem_usage, skip_ops=args.skip_ops, hard_shrink_opt=True)

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    tic = time.time()

    # model.eval()
    for epoch in range(args.start_epoch, args.epochs):
        pseudolossepoch = 0
        lossepoch = 0
        pseudolosscounter = 0
        losscounter = 0

       
        for j, (imgs, imgsjump) in enumerate(zip(train_batch, train_batch_jump)):
            net_in = copy.deepcopy(imgs)
            net_in = net_in.cuda()

            jump_pseudo_stat = []
            cls_labels = []

            for b in range(args.batch_size):
                total_pseudo_prob = 0
                rand_number = np.random.rand()
                pseudo_bool = False

                # skip frame pseudo anomaly
                pseudo_anomaly_jump = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_jump
                total_pseudo_prob += args.pseudo_anomaly_jump
                if pseudo_anomaly_jump:
                    net_in[b] = imgsjump[b][0]
                    jump_pseudo_stat.append(True)
                    pseudo_bool = True
                else:
                    jump_pseudo_stat.append(False)

                if pseudo_bool:
                    cls_labels.append(0)
                else:
                    cls_labels.append(1)

            for b in range(args.batch_size):
                if jump_pseudo_stat[b]:
                    out = model.forward(net_in, mem=False, mem_ano=True)
                else:
                    out = model.forward(net_in, mem=True, mem_ano=False)

            loss_mem = torch.abs(out['mem'].cuda() - out['mem_ano'].cuda()) * -1.0
            loss_mse = loss_func_mse(out["recon"], net_in)

            loss_sparsity = torch.mean(torch.sum(-out["att_weight3"] * torch.log(out["att_weight3"] + 1e-12), dim=1))
            loss_sparsity_ano = -1.0 * torch.mean(
                torch.sum(-out["att_weight3_ano"] * torch.log(out["att_weight3_ano"] + 1e-12), dim=1))

            modified_loss_mse = []
            for b in range(args.batch_size):
                if jump_pseudo_stat[b]:
                    modified_loss_mse.append(torch.mean(-loss_mse[b]))
                    pseudolossepoch += modified_loss_mse[-1].cpu().detach().item()
                    pseudolosscounter += 1

                else:  # no pseudo anomaly
                    modified_loss_mse.append(torch.mean(loss_mse[b]))
                    lossepoch += modified_loss_mse[-1].cpu().detach().item()
                    losscounter += 1
            assert len(modified_loss_mse) == loss_mse.size(0)
            stacked_loss_mse = torch.stack(modified_loss_mse)
            loss_all = (loss_mse + loss_m_weight * loss_sparsity_ano + loss_m_weight * loss_sparsity).sum() + loss_mem.sum() * 0.00002

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        # Save the model and the memory items
        model_dict = {
            'model': model
        }

        torch.save(model_dict, os.path.join(log_dir, 'model_{:02d}.pth'.format(epoch)))
    print('Training is finished')
    toc = time.time()
    print('time:' + str(1000 * (toc - tic)) + "ms")
    sys.stdout = orig_stdout
    f.close()


