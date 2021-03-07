import torch
import torch.optim as optim
import torch.nn.functional as F
from batchers.wiki_s2v_correction_batch import WikiS2vCorrectionBatch
from models.sentence_encoder import SentenceEncoder
from models.sam import SAM
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import sys
from configs import configs
from tqdm import tqdm
import numpy as np
import math

print(int(sys.argv[1]))
config_idx = int(sys.argv[1])
config = configs[int(sys.argv[1])]

if config['training']['log']:
    now = datetime.now()
    writer = [SummaryWriter(log_dir="./train/logs/"+config['name']), SummaryWriter(log_dir="./train/logs/"+config['name']+"_mem0")]

class LabelSmoothingCrossEntropy(torch.nn.Module):
    # based on https://github.com/seominseok0429/label-smoothing-visualization-pytorch
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.2):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss

def train(models, device, train_loader, optimizers, epoch, scheduler=None):
    for model in models:
        model.train()
    train_loss = [0.0]*len(models)
    mem_norm = [0.0]*len(models)

    total = [1e-6]*len(models)
    correct = [0.0]*len(models)

    start = time.time()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    criterion = LabelSmoothingCrossEntropy()
    for batch_idx, (sent, sent_mask, mem_sent, mem_sent_mask, label_sent_, label_sent_mask, label_class) in enumerate(train_loader):

        for idx, _ in enumerate(models):
            sent, sent_mask = sent.to(device), sent_mask.to(device)
            mem_sent, mem_sent_mask = mem_sent, mem_sent_mask # mem_sent will be assing to device after shuffling
            label_sent_, label_sent_mask = label_sent_.to(device), label_sent_mask.to(device)
            label_class = label_class.to(device)
            optimizers[idx].zero_grad()

            if idx%2 == 1:
                mem_sent_ = mem_sent
                # mem_sent_ = torch.cat((mem_sent[1:], mem_sent[0].unsqueeze(0)), dim=0)
                # mem_sent_[:, :, :] = 0.0
                mem_sent_[:, :, :] = torch.randn_like(mem_sent[:, :, :])*2
                # mem_sent_mask[:, :, 1:] = 1.0
            else:
                mem_sent_ = mem_sent
            mem_sent_ = mem_sent_.to(device)
            mem_sent_mask = mem_sent_mask.to(device)
            model_output = models[idx](sent, mem_sent_, label_sent_, sent_mask=sent_mask, mem_sent_mask=mem_sent_mask)
            mem_s2v = model_output['mem_s2v']
            label_sent = model_output['lsent']
            # label_sent = label_sent_
            # pred_class = model_output['pred_class']

            # pred_loss = 0.0
            # for w_idx in range(config['max_sent_len']-1):
            #     pred_loss += torch.sum(criterion(pred_class[:, w_idx], label_class[:, w_idx].to(torch.long), smoothing=0.2) * \
            #                           (1-label_sent_mask[:, w_idx].type(torch.cuda.FloatTensor))) / \
            #                  (torch.sum(1-label_sent_mask[:, w_idx].type(torch.cuda.FloatTensor)) + 1e-6)
                
            #     _, pred_idx = torch.max(pred_class[:, w_idx], 1)
            #     label_idx = label_class[:, w_idx].to(torch.long)
            #     total[idx] += float(torch.sum(1-label_sent_mask[:, w_idx].type(torch.cuda.FloatTensor)))
            #     correct[idx] += float(((pred_idx == label_idx).type(torch.cuda.FloatTensor)*(1-label_sent_mask[:, w_idx].type(torch.cuda.FloatTensor))).sum().item())

            pred_loss = 0
            for i in range(config['training']['num_predictions']):
                sent_pred = model_output['sents'][i]
                ii = i
                if i == 0:
                    ii = -config['max_sent_len']

                pred_loss_pos = torch.sum(torch.mean(torch.pow(sent_pred[:, :-ii] - label_sent[:, i:], 2), dim=2) * \
                                        (1-label_sent_mask[:, i:].type(torch.cuda.FloatTensor))) / \
                                (torch.sum(1-label_sent_mask[:, i:].type(torch.cuda.FloatTensor)) + 1e-6)
                # pred_loss_neg = torch.sum(torch.mean(torch.pow(sent_pred[:, :-i-1] - label_sent[:, i+1:], 2), dim=2) * \
                #                         (1-label_sent_mask[:, i+1:].type(torch.cuda.FloatTensor))) / \
                #                 (torch.sum(1-label_sent_mask[:, i+1:].type(torch.cuda.FloatTensor)) + 1e-6)
                if (batch_idx % 100 == 0):
                    print(pred_loss_pos)
                    # print(pred_loss_neg)
                    print("----")
                pred_loss += pred_loss_pos # - min(pred_loss_neg, 0.6) # torch.nn.functional.threshold(-pred_loss_neg, -0.2, -0.2)
            
            # sent_pred = model_output['sents'][0]
            # pred_loss_neg_in = torch.sum(torch.mean(torch.pow(sent_pred[:, 1:] - label_sent[:, :-1], 2), dim=2) * \
            #                             (1-label_sent_mask[:, :-1].type(torch.cuda.FloatTensor))) / \
            #                 (torch.sum(1-label_sent_mask[:, :-1].type(torch.cuda.FloatTensor)) + 1e-6)
            # pred_loss += - min(pred_loss_neg_in, 0.6)

            mem_norm[idx] += float(torch.mean(torch.norm(mem_s2v, dim=2)))
            # print("-----")
            # print(sent[0,:,0])
            # print(label_sent[0,:,0])
            # print(label_sent_mask[0,:])
            # print(sent_mask[0,:])

            # if (batch_idx == 0):
            if (batch_idx%10 == 0):
                # print(mem_s2v[0:4, 0])
                # print(model_output['sents'][0][0, 5, 0:64])
                # print(model_output['sents'][0][0, 6, 0:64])
                # print("-----------")
                # print(sent[0, 5, 0:64])
                # print(sent[0, 6, 0:64])
                # print(label_sent[0, 5, 0:64])
                # print("---------")
                # print(torch.max(pred_class[0, :], 1))
                # print(label_class[0, :])
                pass

            train_loss[idx] += pred_loss.detach()

            pred_loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(models[idx].parameters(), 0.1)
            optimizers[idx].step()

        # pbar.set_description("L" + str(round(float(pred_loss.detach()), 4)))
        pbar.update(1)
    pbar.close()
    end = time.time()
    for idx, _ in enumerate(models):
        train_loss[idx] /= batch_idx + 1
        mem_norm[idx] /= batch_idx + 1
    if math.isnan(float(train_loss[0])):
        print("nan")
        exit(0)
    print("")
    # print('Epoch {}:\tTrain set: Average loss: {:.4f}'.format(epoch, train_loss))
    print('Epoch {}:'.format(epoch))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer[0].add_scalar('diff/train', train_loss[0]-train_loss[1], epoch)
        for idx, _ in enumerate(models):
            writer[idx].add_scalar('loss/train', train_loss[idx], epoch)
            writer[idx].add_scalar('acc/train', 100*correct[idx]/total[idx], epoch)
            writer[idx].add_scalar('loss/mem_norm', mem_norm[idx], epoch)
            writer[idx].flush()

def test(models, device, test_loader, epoch):
    for model in models:
        model.eval()
    test_sent_loss = []
    total = [1e-6]*len(models)
    correct = [0.0]*len(models)
    for idx, _ in enumerate(models):
        test_sent_loss.append([0.0]*config['training']['num_predictions'])
    criterion = LabelSmoothingCrossEntropy()
    with torch.no_grad():
        for batch_idx, (sent, sent_mask, mem_sent, mem_sent_mask, label_sent_, label_sent_mask, label_class) in enumerate(test_loader):
            for idx, _ in enumerate(models):
                sent, sent_mask = sent.to(device), sent_mask.to(device)
                mem_sent, mem_sent_mask = mem_sent, mem_sent_mask # mem_sent will be assing to device after shuffling
                label_sent_, label_sent_mask = label_sent_.to(device), label_sent_mask.to(device)
                # next_sent, next_sent_mask = next_sent.to(device), next_sent_mask.to(device)
                label_class = label_class.to(device)

                if idx%2 == 1:
                    mem_sent_ = mem_sent
                    # mem_sent_ = torch.cat((mem_sent[1:], mem_sent[0].unsqueeze(0)), dim=0)
                    # mem_sent_[:, :, :] = 0.0
                    mem_sent_[:, :, :] = torch.randn_like(mem_sent[:, :, :])*2
                    # mem_sent_mask[:, :, :] = 1.0
                else:
                    mem_sent_ = mem_sent
                mem_sent_ = mem_sent_.to(device)
                mem_sent_mask = mem_sent_mask.to(device)
                model_output = models[idx](sent, mem_sent_, label_sent_, sent_mask=sent_mask, mem_sent_mask=mem_sent_mask)
                label_sent = model_output['lsent']
                # label_sent = label_sent_
                # pred_class = model_output['pred_class']

                # for w_idx in range(config['max_sent_len']-1):
                #     test_sent_loss[idx][0] += torch.sum(criterion(pred_class[:, w_idx], label_class[:, w_idx].to(torch.long), smoothing=0.2) * \
                #                         (1-label_sent_mask[:, w_idx].type(torch.cuda.FloatTensor))) / \
                #                 (torch.sum(1-label_sent_mask[:, w_idx].type(torch.cuda.FloatTensor)) + 1e-6)

                #     _, pred_idx = torch.max(pred_class[:, w_idx], 1)
                #     label_idx = label_class[:, w_idx].to(torch.long)
                #     total[idx] += float(torch.sum(1-label_sent_mask[:, w_idx].type(torch.cuda.FloatTensor)))
                #     correct[idx] += float(((pred_idx == label_idx).type(torch.cuda.FloatTensor)*(1-label_sent_mask[:, w_idx].type(torch.cuda.FloatTensor))).sum().item())

                for i in range(config['training']['num_predictions']):
                    sent_pred = model_output['sents'][i]
                    ii = i
                    if i == 0:
                        ii = -config['max_sent_len']
                    pred_loss_pos = torch.sum(torch.mean(torch.pow(sent_pred[:, :-ii] - label_sent[:, i:], 2), dim=2) * \
                                        (1-label_sent_mask[:, i:].type(torch.cuda.FloatTensor))) / \
                                    (torch.sum(1-label_sent_mask[:, i:].type(torch.cuda.FloatTensor)) + 1e-6)
                    # pred_loss_neg = torch.sum(torch.mean(torch.pow(sent_pred[:, :-i-1] - label_sent[:, i+1:], 2), dim=2) * \
                    #                     (1-label_sent_mask[:, i+1:].type(torch.cuda.FloatTensor))) / \
                    #                 (torch.sum(1-label_sent_mask[:, i+1:].type(torch.cuda.FloatTensor)) + 1e-6)
                    # # test_sent_loss[idx][i] += pred_loss_pos - torch.nn.functional.threshold(-pred_loss_neg, -0.2, -0.2)
                    test_sent_loss[idx][i] += pred_loss_pos # - min(pred_loss_neg, 0.6) # torch.nn.functional.threshold(-pred_loss_neg, -0.2, -0.2)

    for idx, _ in enumerate(models):
        for i in range(config['training']['num_predictions']):
            test_sent_loss[idx][i] /= batch_idx + 1
    if config['training']['log']:
        for i in range(config['training']['num_predictions']):
            writer[0].add_scalar('diff/test_sent_'+str(i), test_sent_loss[0][i]-test_sent_loss[1][i], epoch)
        for idx, _ in enumerate(models):
            for i in range(config['training']['num_predictions']):
                writer[idx].add_scalar('loss/test_sent_'+str(i), test_sent_loss[idx][i], epoch)
                writer[idx].add_scalar('acc/test', 100*correct[idx]/total[idx], epoch)
            writer[idx].flush()
    return test_sent_loss[0][0]

def sentence_score_prediction(models, device, dataset):
    model.eval()
    title_idx = 5
    with torch.no_grad():
        memory_sentences_score = {}
        test_sentences, mem_sentences = dataset.get_sentences_from_doc(title_idx)
        scores = []
        for i in range(len(mem_sentences)):
            score = []
            for ii in range(len(mem_sentences)):
                score.append(0)
            scores.append(score)

        pbar = tqdm(total=len(test_sentences), dynamic_ncols=True)
        for test_sent in test_sentences:
            for mem_sent in mem_sentences:
                sent, sent_mask = test_sent['emb'].to(device), test_sent['mask'].to(device)
                label_sent, label_sent_mask = test_sent['label'].to(device), test_sent['label_mask'].to(device)
                mem_sent_, mem_sent_mask = mem_sent['emb'].to(device), mem_sent['mask'].to(device)

                model_output = models[0](sent, mem_sent_, None, sent_mask=sent_mask, mem_sent_mask=mem_sent_mask)
                sent_pred = model_output['sents'][0]
                pred_loss_pos = torch.sum(torch.mean(torch.pow(sent_pred - label_sent, 2), dim=2) * \
                                    (1-label_sent_mask.type(torch.cuda.FloatTensor))) / \
                                torch.sum(1-label_sent_mask.type(torch.cuda.FloatTensor))
                pred_loss = pred_loss_pos

                # no memory
                sent, sent_mask = test_sent['emb'].to(device), test_sent['mask'].to(device)
                label_sent, label_sent_mask = test_sent['label'].to(device), test_sent['label_mask'].to(device)
                mem_sent_, mem_sent_mask = torch.clone(mem_sent['emb']), mem_sent['mask'].to(device)
                mem_sent_[:, :, :] = torch.randn_like(mem_sent_[:, :, :])*2
                mem_sent_ = mem_sent_.to(device)

                model_output = models[1](sent, mem_sent_, None, sent_mask=sent_mask, mem_sent_mask=mem_sent_mask)
                sent_pred = model_output['sents'][0]
                pred_loss_pos = torch.sum(torch.mean(torch.pow(sent_pred - label_sent, 2), dim=2) * \
                                    (1-label_sent_mask.type(torch.cuda.FloatTensor))) / \
                                torch.sum(1-label_sent_mask.type(torch.cuda.FloatTensor))
                pred_loss_nmem = pred_loss_pos

                scores[mem_sent['idx']][test_sent['idx']] = float(pred_loss_nmem) - float(pred_loss)
                # print("----------------")
                # print(mem_sent['idx'], test_sent['idx'])
                # print("test:\t", test_sent['text'])
                # print(mem_sent['text'])
                # print("mem:\t", mem_sent['text'])
                # print(float(pred_loss_nmem)-float(pred_loss))
                # if mem_sent['idx'] not in memory_sentences_score:
                #     memory_sentences_score[mem_sent['idx']] = {'score': float(pred_loss_nmem)-float(pred_loss), 'text': mem_sent['text']}
                # else:
                #     if not math.isnan(float(pred_loss)):
                #         memory_sentences_score[mem_sent['idx']]['score'] += float(pred_loss_nmem)-float(pred_loss)
            # break
            pbar.update(1)
        # max_score = -1e6
        # min_score = 1e6
        for i in range(len(scores)):
            for ii in range(len(scores[i])):
                print(scores[i][ii], end='\t')
            print("")
        # for sentence in memory_sentences_score.values():
        #     if sentence['score'] > max_score:
        #         max_score = sentence['score']
        #     if sentence['score'] < min_score:
        #         min_score = sentence['score']
        # for sentence in memory_sentences_score.values():
        #     # norm_score = round(100-(sentence['score']-min_score)/(max_score-min_score)*100)
        #     norm_score = sentence['score']
        #     print(str(norm_score) + "\t" + sentence['text'])
    exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = SentenceEncoder(config)
restore_name = '_b24sL48_Adamlr8e-05s10g0.97_memTrue=-1.cnt1.g0.0dr0.0.poolmean.s2v1024_mTr4idr0.0.mha16.ffn2048.hdr0.0.gT.fi.mha.nPred1_v92_inLLoutLL_sent+-3_s2vGTrx0.mhaPool.nGate.nNorm_mTrHhaH1k_trD40_maskedSentLLayDr.3othDocFix.1Zero_rndMem_lossFullSent_616'
checkpoint = torch.load('./train/save/'+restore_name)
model.load_state_dict(checkpoint['model_state_dict'])
print(model)
model.to(device)
start_epoch = 1
start_epoch += checkpoint['epoch']
del checkpoint

model_nmem = SentenceEncoder(config)
checkpoint = torch.load('./train/save/'+restore_name+'_mem0')
model_nmem.load_state_dict(checkpoint['model_state_dict'])
model_nmem.to(device)
del checkpoint

dataset_train = WikiS2vCorrectionBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=4)
dataset_test = WikiS2vCorrectionBatch(config, valid=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=config['batch_size'],
    shuffle=False, num_workers=0)

sentence_score_prediction([model, model_nmem], device, dataset_test)

optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
optimizer_nmem = optim.Adam(model_nmem.parameters(), lr=config['training']['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['lr_step'],
                                                       gamma=config['training']['lr_gamma'])
scheduler_nmem = torch.optim.lr_scheduler.StepLR(optimizer_nmem, step_size=config['training']['lr_step'],
                                                                 gamma=config['training']['lr_gamma'])
test_loss = 1e6
for epoch in range(start_epoch, config['training']['epochs'] + start_epoch):
    train([model, model_nmem], device, data_loader_train, [optimizer, optimizer_nmem], epoch, None)
    current_test_loss = test([model, model_nmem], device, data_loader_test, epoch)
    dataset_train.on_epoch_end()
    if current_test_loss < test_loss:
        test_loss = current_test_loss
        print("mean="+str(test_loss))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': test_loss
            }, './train/save/'+config['name'])
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_nmem.state_dict(),
            'loss': test_loss
            }, './train/save/'+config['name']+'_mem0')
    scheduler_nmem.step()
    scheduler.step()

if config['training']['log']:
    writer[0].close()
    writer[1].close()
