# Multi_GPU
import os
import time
import os.path as osp
from os.path import exists
import argparse
import json
import logging
import time
import copy
import numpy as np
import segmentation_models_pytorch as smp
##### pytorch library #####
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
##multi GPU use
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data as data
##### My own library #####
import data.seg_transforms as dt
from data.seg_dataset import segList
from models.net_builder import net_builder
from loss import ch_loss_builder2


def last_checkpoint(output):
    import glob

    def corrupted(fpath):
        try:
            torch.load(fpath, map_location='cpu')
            return False
        except:
            print(f'WARNING: Cannot load {fpath}')
            return True

    saved = sorted(
        glob.glob(f'{output}/checkpoint_*.pt'),
        key=lambda f: int(re.search('_(\d+).pt', f).group(1)))

    if len(saved) >= 1 and not corrupted(saved[-1]):
        return saved[-1]
    elif len(saved) >= 2:
        return saved[-2]
    else:
        return None

def save_checkpoint(model, optimizer,scaler, args, model_config, data_config, epoch, total_iter, loss):    
    import os
    import shutil    
    
    if args.local_rank != 0:
        return

    if args.fp16:
        if args.amp == 'pytorch':
            amp_state = scaler.state_dict()
        elif args.amp == 'apex':
            amp_state = amp.state_dict()
    else:
        amp_state = None
    
    state = {
        'args'           : args,
        'model_config'   : model_config,
        'data_config'    : data_config,        
        'model_state'    : model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'amp_state'      : amp_state,
        'epoch'          : epoch,
        'total_iter'     : total_iter,
        'val_loss'       : loss,
        }

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # filepath = os.path.join(args.output_dir, 'checkpoint_{}.pt'.format(epoch) )
    epoch_str = "{:05d}".format(epoch)
    filepath = os.path.join(args.output_dir, 'checkpoint_{}.pt'.format(epoch_str) )
    torch.save(state, filepath)   

    last_chkpt_filepath = os.path.join(args.output_dir, 'checkpoint_last.pt')
    shutil.copy(filepath, last_chkpt_filepath)


def load_checkpoint(args, model, optimizer, scaler, start_epoch, start_iter):
    import os
    path = os.path.join(args.output_dir, 'checkpoint_last.pt')
    dst = f'cuda:{torch.cuda.current_device()}'
    checkpoint = torch.load(path, map_location=dst)
   
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if args.fp16:
        if args.amp == 'pytorch':
            scaler.load_state_dict(checkpoint['amp_state'])
        elif args.amp == 'apex':
            amp.load_state_dict(checkpoint['amp_state'])
    #args         = checkpoint['args']
    #model_config = checkpoint['model_config']
    #data_config  = checkpoint['data_config']
    #data_config  = checkpoint['data_config']
    val_loss     = checkpoint['val_loss']    
    start_epoch[0] = checkpoint['epoch'] + 1
    start_iter[0] = checkpoint['total_iter']
    if args.local_rank ==0:
        print("\nDEBUG : {} : load {} and resume epoch {:d} total iter : {:d} with loss {}".format(args.local_rank, path, start_epoch[0], start_iter[0]+1, val_loss  ) )

def init_distributed(args, world_size, rank):
    import torch
    import torch.distributed as dist
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing distributed training")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    backend = 'nccl' #if args.cuda else 'gloo'
    dist.init_process_group(backend=backend,
                            init_method='env://')
    print("Done initializing distributed training")


def parse_args():
    parser = argparse.ArgumentParser(description='train')
    # config
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('--name', dest='name',help='change model',default=None, type=str)
    parser.add_argument('-j', '--workers', type=int, default=16)
    # train setting
    parser.add_argument('--step', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--t', type=str, default='t1')
    parser.add_argument('--model-path', help='pretrained model test', default=' ', type=str)
    #multi gpu
    parser.add_argument('--local_rank',type=int, default=os.getenv('LOCAL_RANK', 0),  help='Rank of the process for multiproc. Do not set manually.')
    parser.add_argument('--rank',type=int, default=os.getenv('RANK', 0),  help='Rank of the process for multiproc. Do not set manually.')
    parser.add_argument('--world_size',type=int, default=os.getenv('WORLD_SIZE', 1),  help='Number of processes for multiproc. Do not set manually.')  
    parser.add_argument('--seed',type=int, default=1234,help='Seed for PyTorch random number generators')
    parser.add_argument('--multi_gpu',type=str,default='ddp',choices=['ddp', 'dp'],help='Use multiple GPU')
    #chckpoint
    parser.add_argument('--fp16', action='store_true',help='Run training in fp16/mixed precision')
    parser.add_argument('--amp',type=str,default='pytorch', choices=['apex', 'pytorch'], help='Implementation of automatic mixed precision')
    parser.add_argument('--resume',action='store_true',help='Resume training from the last available checkpoint') 
    parser.add_argument('-o', '--output-dir', type=str, default='./output', help='Directory to save checkpoints')
    parser.add_argument('--sample-dir', type=str, default='./samples', help='Directory to save checkpoints') 
    parser.add_argument('--checkpoint-path',  type=str, default=None,     help='Checkpoint path to resume training')
    parser.add_argument( '--clip', type=float, default=0.25,   help='Clip threshold for gradients')    
    parser.add_argument( '--grad-clip-thresh', type=float, default=1000.0,   help='Clip threshold for gradients')
    
    args = parser.parse_args()
    return args

def main():
    ##### config #####
    args = parse_args()
    total_iter = 0
    # **single    
    # seed = 1234
    # torch.manual_seed(seed)
    # orch.cuda.manual_seed(seed)
    device = torch.device('cuda')
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    init_distributed(args, args.world_size, args.local_rank)
    print('local_rank:',args.local_rank)
    if args.local_rank==0:
        print('torch version:',torch.__version__)
    
    
    ##### result path setting #####
    tn = args.t 
    task_name = args.data_dir.split('/')[-2] + '/' + args.data_dir.split('/')[-1]
    
    train_result_path = osp.join('result',task_name,'train',args.name + '_' +str(args.lr) + '_'+ tn)
    test_result_path = osp.join('result',task_name,'test',args.name + '_' +str(args.lr) + '_'+ tn)
    pred_result_path = osp.join('result',task_name,'predict',args.name + '_' +str(args.lr) + '_'+ tn)
    if args.local_rank==0:
        if not exists(train_result_path):
            os.makedirs(train_result_path)
        if not exists(test_result_path):
            os.makedirs(test_result_path)
        if not exists(pred_result_path):
            os.makedirs(pred_result_path)
    
    # print hyperparameters
    if args.local_rank==0:    
        for k, v in args.__dict__.items():
            print(k, ':', v)
            
    # define loss function
    criterion2 = ch_loss_builder2()
    
    # load the network
#     model = net_builder(args.name)
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=1,
        classes=8
    )
    
    model.to(device)
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), #Adam optimizer
                                    args.lr,
                                    betas=(0.9, 0.99),
                                    weight_decay=args.weight_decay) 
    ### configure AMP    
    if args.local_rank==0:
        print("DEBUG : AMP config" )
    scaler = None    
    if args.fp16 and args.amp == 'pytorch':
            scaler = torch.cuda.amp.GradScaler()
            
    if args.multi_gpu == 'ddp':
        para_model = DDP(model, device_ids=[args.local_rank],output_device=args.local_rank,
                                broadcast_buffers=False,find_unused_parameters=False, )
        distributed_run = True
    else:
        para_model = model
    
    start_epoch = [1]
    start_iter  = [0]
           
    if args.resume:
        load_checkpoint(args, model, optimizer, scaler, start_epoch, start_iter)#load scaler as well

    start_epoch = start_epoch[0]
    total_iter = start_iter[0]

    if args.local_rank==0:   
        print('#'*15,args.name,'#'*15)

    ##### load dataset #####
    info = json.load(open(osp.join(args.data_dir, 'info.json'), 'r'))
    normalize = dt.Normalize(mean=info['mean'], std=info['std'])
    t = []
    t.extend([dt.Label_Transform(),dt.ToTensor(),normalize])
    train_dataset = segList(args.data_dir, 'train', dt.Compose(t))
    val_dataset = segList(args.data_dir, 'eval', dt.Compose(t))
    test_dataset = segList(args.data_dir, 'test', dt.Compose(t))
    pred_dataset = segList(args.data_dir, 'predict', dt.Compose(t))
    print("train dataset:",len(train_dataset))
    #Multi gpu sampler
    if  args.multi_gpu == 'ddp' and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(val_dataset)     
        test_sampler = DistributedSampler(test_dataset)
        pred_sampler = DistributedSampler(pred_dataset) 
        shuffle=False
    else :
        train_sampler=None
        eval_sampler=None
        test_sampler=None
        pred_sampler=None
        shuffle=False        
    print("train sampler:",train_sampler)
    #Loader
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,sampler=train_sampler, shuffle=shuffle, num_workers=args.workers, pin_memory=True, drop_last=True)
    eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,sampler=eval_sampler, shuffle=shuffle, num_workers=args.workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler,shuffle=shuffle, num_workers=args.workers, pin_memory=False)
    pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=1, sampler=pred_sampler,shuffle=shuffle, num_workers=0, pin_memory=False)

    if args.local_rank ==0:
        print("DEBUG : train_loader : ",  len(train_loader) )
        print("DEBUG : valid_loader : ",  len(eval_loader) )


    
    for epoch in range(start_epoch, args.epochs+1):
        ch_epoch_loss = 0.
        model.train()
        model.zero_grad()
        tick_epochs = time.time()
        for i,(input,target) in enumerate(train_loader):
            tick_iter=time.time()

            # variable
            input_var = Variable(input).cuda()
            target_var_seg = Variable(target).cuda()
            
            enable_autocast = args.fp16 and args.amp == 'pytorch'
            
            with torch.cuda.amp.autocast(enable_autocast):    
                # forward
                output_seg = para_model(input_var)
                
                #loss_NLL = criterion2[0](output_seg, target_var_seg)
                loss_DICELOSS = criterion2(output_seg, target_var_seg)#[1](output_seg, target_var_seg)
                loss = loss_DICELOSS     # loss from the two-stage network   
                ch_epoch_loss += loss
                loss.requires_grad_(True)#added
                    
            #AMP loss
            if args.fp16 and args.amp == 'pytorch':
                    scaler.scale(loss).backward()

            else:
                loss.backward()

            ## gradient clipping
            if args.fp16 and  args.amp == 'pytorch':
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                
            ## scaler
            if args.fp16 and args.amp == 'pytorch':
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()   
            
            tock_iter = time.time()
            duration_iter = tock_iter-tick_iter
            total_iter += 1
            if args.local_rank==0:
                str_iter="epoch: {} iter: {}/{} Loss:{} duration:{}ms".format(epoch,i,len(train_loader),loss.item(),duration_iter)
                print(str_iter)
                if i+1 == len(train_loader):
                    print_loss = ch_epoch_loss.item()/len(train_loader)
                    print(f"AVERAGE LOSS for epoch {epoch}: {print_loss}")
                    ch_epoch_loss = 0.0
                    print("\n")
                
        duration_iter_1 = tock_iter - tick_epochs 
        if args.local_rank==0:
            print("duration_epochs:",duration_iter_1)
            loss_list = {'dice':print_loss, 'total':loss}
            model_config=None
            data_config=None
            save_checkpoint(model, optimizer,scaler, args, model_config, data_config, epoch, total_iter, loss_list)  

if __name__ == '__main__':
    main()
