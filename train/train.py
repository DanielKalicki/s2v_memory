import torch
import torch.optim as optim
import torch.nn.functional as F
from batchers.wiki_s2v_correction_batch import WikiS2vCorrectionBatch
from models.sentence_encoder import SentenceEncoder
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import sys
from configs import configs
from tqdm import tqdm

print(int(sys.argv[1]))
config = configs[int(sys.argv[1])]

if config['training']['log']:
    now = datetime.now()
    writer = SummaryWriter(log_dir="./train/logs/"+config['name'])

def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    train_loss = 0.0
    start = time.time()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    # criterion = torch.nn.MSELoss()
    for batch_idx, (sent, sent_mask, mem_sent, mem_sent_mask, label_sent, label_sent_mask) in enumerate(train_loader):
        sent, sent_mask = sent.to(device), sent_mask.to(device)
        mem_sent, mem_sent_mask = mem_sent.to(device), mem_sent_mask.to(device)
        label_sent, label_sent_mask = label_sent.to(device), label_sent_mask.to(device)
        optimizer.zero_grad()

        model_output = model(sent, mem_sent, sent_mask=sent_mask, mem_sent_mask=mem_sent_mask)
        sent_pred = model_output['sent']

        # pred_loss = criterion(sent_pred, label_sent)
        sent_pred = torch.nn.functional.normalize(sent_pred, dim=2)
        label_sent = torch.nn.functional.normalize(label_sent, dim=2)
        pred_loss = torch.sum(torch.mean(torch.pow(sent_pred - label_sent, 2), dim=2) * \
                             (1-label_sent_mask.type(torch.cuda.FloatTensor))) / \
                    torch.sum(1- label_sent_mask.type(torch.cuda.FloatTensor))
        train_loss += pred_loss.detach()

        pred_loss.backward(retain_graph=True)
        optimizer.step()

        if scheduler:
            scheduler.step()

        pbar.set_description("L" + str(round(float(pred_loss.detach()), 4)))
        pbar.update(1)
    pbar.close()
    end = time.time()
    train_loss /= batch_idx + 1
    print("")
    print('Epoch {}:\tTrain set: Average loss: {:.4f}'.format(epoch, train_loss))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.flush()

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0.0
    # criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (sent, sent_mask, mem_sent, mem_sent_mask, label_sent, label_sent_mask) in enumerate(test_loader):
            sent, sent_mask = sent.to(device), sent_mask.to(device)
            mem_sent, mem_sent_mask = mem_sent.to(device), mem_sent_mask.to(device)
            label_sent, label_sent_mask = label_sent.to(device), label_sent_mask.to(device)

            model_output = model(sent, mem_sent, sent_mask=sent_mask, mem_sent_mask=mem_sent_mask)
            sent_pred = model_output['sent']

            # pred_loss = criterion(sent_pred, label_sent)
            sent_pred = torch.nn.functional.normalize(sent_pred, dim=2)
            label_sent = torch.nn.functional.normalize(label_sent, dim=2)
            pred_loss = torch.sum(torch.mean(torch.pow(sent_pred - label_sent, 2), dim=2) * \
                                 (1-label_sent_mask.type(torch.cuda.FloatTensor))) / \
                        torch.sum(1- label_sent_mask.type(torch.cuda.FloatTensor))
            test_loss += pred_loss.detach()

    test_loss /= batch_idx + 1
    if config['training']['log']:
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.flush()
    print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    return test_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = SentenceEncoder(config)
# restore_name = 'b24sL32_Adamlr8e-05s10g0.5_gTr4.mha16.ffn256.poolmean.s2v2048_mTr4idr0.0.mha16.ffn256.postNorm_memTrue=sent.onlyMaskLoss_inMaskW0_v5_0'
# checkpoint = torch.load('./train/save/'+restore_name)
# model.load_state_dict(checkpoint['model_state_dict'])
print(model)
model.to(device)
start_epoch = 1
# start_epoch += checkpoint['epoch']
# del checkpoint

dataset_train = WikiS2vCorrectionBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=0)
dataset_test = WikiS2vCorrectionBatch(config, valid=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=config['batch_size'],
    shuffle=False, num_workers=0)

if config['training']['optimizer'] == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['lr_step'],
                                                       gamma=config['training']['lr_gamma'])
test_loss = 1e6
for epoch in range(start_epoch, config['training']['epochs'] + start_epoch):
    train(model, device, data_loader_train, optimizer, epoch, None)
    current_test_loss = test(model, device, data_loader_test, epoch)
    dataset_train.on_epoch_end()
    if current_test_loss < test_loss:
        test_loss = current_test_loss
        print("mean="+str(test_loss))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss
            }, './train/save/'+config['name'])
    scheduler.step()

if config['training']['log']:
    writer.close()
