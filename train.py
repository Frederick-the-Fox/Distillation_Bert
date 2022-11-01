import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data as Data
from data_process import load_data
from model import WangYC_Model
import os

def train(args, writer):

    # DDP参数初始化
    # local_rank = args.local_rank
    torch.cuda.set_device(args.local_rank)

    # DDP后端初始化
    # device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend='nccl')

    print("start loading data...")
    train_inputs, train_targets = load_data('train')

    print("data loaded! ")
    # epochs = args.epoch
    # batch_size = args.batch_size
    # data_dir = args.data_dir
    epochs = args.epoch
    batch_size = args.batch_size

    # DDP利用随机种子采样的封装好的方法
    train_sampler = torch.utils.data.distributed.DistributedSampler

    train_sentence_loader = Data.DataLoader(
        dataset=train_inputs,
        batch_size=batch_size,  # 每块的大小
    )
    train_label_loader = Data.DataLoader(
        dataset=train_targets,
        batch_size=batch_size,
    )
    
    model = WangYC_Model()
    model = model.to(args.local_rank)
    #DDP 包装model
    if torch.distributed.get_rank() == 0 and os.path.exists(args.model_save + '/saved.pkl'):
        model.load_state_dict(torch.load(args.model_save))
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    #用DDP包装后的model的参数来初始化迭代器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = torch.nn.MSELoss(reduction='mean')
    criterion = criterion.to(args.local_rank)

    print("training start...")
    model.train()

    iteration = 0

    for epoch in range(epochs): # 开始训练
        print('...this is epoch : {}...'.format(epoch))
        loss_list = []       
        for sentences, labels in zip(train_sentence_loader, train_label_loader):
            sentences = sentences
            labels = labels.to(args.local_rank)
            # print (labels)
            # break
            optimizer.zero_grad()
            result = model(sentences)
            # print(result)
            # print(labels)
            loss = torch.tensor(0)
            loss_ite = 0
            best_eval = 0
            es_count = 0
            for each_result in result:
                # print('labels:{}'.format(labels[0][loss_ite]))
                # print('each_result:{}'.format(each_result))
                # print('labels[loss_ite]:{}'.format(labels[loss_ite]))
                # print('labels[loss_ite].detach().numpy():{}'.format(labels[loss_ite].detach().numpy()[0][0]))
                if labels[0].cpu().detach().numpy()[loss_ite][0] == 1:
                    loss = loss + args.weight * criterion(each_result, labels[0][loss_ite])
                else:
                    loss = loss + criterion(each_result, labels[0][loss_ite])
                loss_ite += 1
            
            loss = loss / args.weight
            print('this is iteration : {} and loss : {}'.format(iteration, loss.item()))
            writer.add_scalar("loss",loss,iteration)
            loss.backward()
            optimizer.step()
            # break
            loss_list.append(loss.cpu().detach().numpy())
            iteration += 1
            if iteration and iteration % args.val_per_ite == 0:
                torch.save(model.module.state_dict(), args.model_save + '/saved.pkl')
                model.eval()
                acc, sentence_acc = eval(args)
                writer.add_scalar("binary_acc",acc,iteration)
                writer.add_scalar("sentence_acc",sentence_acc,iteration)
                if acc >= best_eval:
                    best_eval = acc
                    es_count = 0
                else:
                    es_count += 1
                
                if es_count == 30:
                    print('early_stopping!!!')
                    break
        
        if torch.distributed.get_rank() == 0:
            torch.save(model.module.state_dict(), args.model_save + '/final.pkl')