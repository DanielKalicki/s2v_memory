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
    writer = SummaryWriter(log_dir="./train/logs/"+config['name'])

def pretrain(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    pretrain_loss = 0.0

    start = time.time()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    for batch_idx, (sent, sent_mask, mem_sent, mem_sent_mask, label_sent, label_sent_mask, next_sent, next_sent_mask) in enumerate(train_loader):
        sent, sent_mask = sent.to(device), sent_mask.to(device)
        mem_sent, mem_sent_mask = mem_sent.to(device), mem_sent_mask.to(device)
        label_sent, label_sent_mask = label_sent.to(device), label_sent_mask.to(device)
        next_sent, next_sent_mask = next_sent.to(device), next_sent_mask.to(device)
        optimizer.zero_grad()

        model_output = model(sent, mem_sent, next_sent, sent_mask=sent_mask, mem_sent_mask=mem_sent_mask, nsent_mask=next_sent_mask)
        mem_s2v = model_output['mem_s2v'][:, 0]

        drop_mask = torch.from_numpy(
                np.random.binomial(1, 1-0.3, (config['word_edim'])).astype(np.float32))
        mem_sent = mem_sent*drop_mask.to(device)
        model_output = model(sent, mem_sent, next_sent, sent_mask=sent_mask, mem_sent_mask=mem_sent_mask, nsent_mask=next_sent_mask)
        mem_s2v_masked = model_output['mem_s2v'][:, 0]
        
        pred_loss_neg = torch.sum(torch.mean(torch.pow(mem_s2v[:-1] - mem_s2v[1:], 2), dim=1))
        pred_loss_pos = torch.sum(torch.mean(torch.pow(mem_s2v - mem_s2v_masked, 2), dim=1))
        pred_loss = pred_loss_pos - torch.nn.functional.threshold(-pred_loss_neg, -0.2, -0.2)

        pretrain_loss += pred_loss.detach()

        pred_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        if batch_idx == 0:
            print(mem_s2v[0])
            print(mem_s2v_masked[0])
            print(mem_s2v[1])

        if scheduler:
            scheduler.step()

        pbar.set_description("L" + str(round(float(pred_loss.detach()), 4)))
        pbar.update(1)
    pbar.close()
    end = time.time()
    pretrain_loss /= batch_idx + 1
    print("")
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('loss/pretrain', pretrain_loss, epoch)
        writer.flush()

def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    train_loss = 0.0

    start = time.time()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    for batch_idx, (sent, sent_mask, mem_sent, mem_sent_mask, label_sent, label_sent_mask, next_sent, next_sent_mask) in enumerate(train_loader):
        sent, sent_mask = sent.to(device), sent_mask.to(device)
        mem_sent, mem_sent_mask = mem_sent.to(device), mem_sent_mask.to(device)
        label_sent, label_sent_mask = label_sent.to(device), label_sent_mask.to(device)
        next_sent, next_sent_mask = next_sent.to(device), next_sent_mask.to(device)
        optimizer.zero_grad()

        model_output = model(sent, mem_sent, next_sent, sent_mask=sent_mask, mem_sent_mask=mem_sent_mask, nsent_mask=next_sent_mask)
        mem_s2v = model_output['mem_s2v']

        pred_loss = 0
        for i in range(config['training']['num_predictions']):
            sent_pred = model_output['sents'][i]
            ii = i
            if i == 0:
                ii = -config['max_sent_len']
            pred_loss_pos = torch.sum(torch.mean(torch.pow(sent_pred[:, :-ii] - label_sent[:, i:], 2), dim=2) * \
                                     (1-label_sent_mask[:, i:].type(torch.cuda.FloatTensor))) / \
                            torch.sum(1-label_sent_mask[:, i:].type(torch.cuda.FloatTensor))
            pred_loss_neg = torch.sum(torch.mean(torch.pow(sent_pred[:, :-i-1] - label_sent[:, i+1:], 2), dim=2) * \
                                     (1-label_sent_mask[:, i+1:].type(torch.cuda.FloatTensor))) / \
                            torch.sum(1-label_sent_mask[:, i+1:].type(torch.cuda.FloatTensor))
            pred_loss += pred_loss_pos - torch.nn.functional.threshold(-pred_loss_neg, -0.02, -0.02)

        # s2v_norm_loss = torch.mean(torch.pow((torch.norm(sent_pred, dim=2)-torch.norm(label_sent, dim=2))/20.0, 4))
        # mem_norm_loss = torch.mean(torch.pow((torch.norm(mem_s2v, dim=2)-13.0)/20.0, 4)) # 13 => norm of sent vectors
        # pred_loss += mem_norm_loss + s2v_norm_loss

        train_loss += pred_loss.detach()

        pred_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        if batch_idx == 0:
            print("mem_s2v norm = " + str(float(torch.mean(torch.norm(mem_s2v, dim=2)))))
            print("label_sent norm = " + str(float(torch.mean(torch.norm(label_sent, dim=2)))))
            print("sent_pred norm = " + str(float(torch.mean(torch.norm(sent_pred, dim=2)))))

        if scheduler:
            scheduler.step()

        pbar.set_description("L" + str(round(float(pred_loss.detach()), 4)))
        pbar.update(1)
    pbar.close()
    end = time.time()
    train_loss /= batch_idx + 1
    if math.isnan(float(train_loss)):
        exit(0)
    print("")
    print('Epoch {}:\tTrain set: Average loss: {:.4f}'.format(epoch, train_loss))
    print('\t\tTraining time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.flush()

def test(model, device, test_loader, epoch):
    model.eval()
    # test_loss = 0.0
    # test_s2_loss = 0.0
    # test_nsent_loss = 0.0
    test_sent_loss = [0.0]*config['training']['num_predictions']
    with torch.no_grad():
        for batch_idx, (sent, sent_mask, mem_sent, mem_sent_mask, label_sent, label_sent_mask, next_sent, next_sent_mask) in enumerate(test_loader):
            sent, sent_mask = sent.to(device), sent_mask.to(device)
            mem_sent, mem_sent_mask = mem_sent.to(device), mem_sent_mask.to(device)
            label_sent, label_sent_mask = label_sent.to(device), label_sent_mask.to(device)
            next_sent, next_sent_mask = next_sent.to(device), next_sent_mask.to(device)

            model_output = model(sent, mem_sent, next_sent, sent_mask=sent_mask, mem_sent_mask=mem_sent_mask, nsent_mask=next_sent_mask)

            for i in range(config['training']['num_predictions']):
                sent_pred = model_output['sents'][i]
                ii = i
                if i == 0:
                    ii = -config['max_sent_len']
                pred_loss_pos = torch.sum(torch.mean(torch.pow(sent_pred[:, :-ii] - label_sent[:, i:], 2), dim=2) * \
                                    (1-label_sent_mask[:, i:].type(torch.cuda.FloatTensor))) / \
                                torch.sum(1-label_sent_mask[:, i:].type(torch.cuda.FloatTensor))
                pred_loss_neg = torch.sum(torch.mean(torch.pow(sent_pred[:, :-i-1] - label_sent[:, i+1:], 2), dim=2) * \
                                    (1-label_sent_mask[:, i+1:].type(torch.cuda.FloatTensor))) / \
                                torch.sum(1-label_sent_mask[:, i+1:].type(torch.cuda.FloatTensor))
                test_sent_loss[i] += pred_loss_pos - torch.nn.functional.threshold(-pred_loss_neg, -0.02, -0.02)

    for i in range(config['training']['num_predictions']):
        test_sent_loss[i] /= batch_idx + 1
    if config['training']['log']:
        for i in range(config['training']['num_predictions']):
            writer.add_scalar('loss/test_sent_'+str(i), test_sent_loss[i], epoch)
        writer.flush()
    # print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    return test_sent_loss[0]

def sentence_score_prediction(model, device, dataset):
    model.eval()
    title_idx = 7
    with torch.no_grad():
        memory_sentences_score = {}
        document_sentences = dataset.get_sentences_from_doc(title_idx)
        pbar = tqdm(total=len(document_sentences), dynamic_ncols=True)
        for input_sent in document_sentences:
            for doc_sent in document_sentences:
                if doc_sent['idx'] != input_sent['idx']:
                    sent, sent_mask = input_sent['emb'][:, 0].to(device), input_sent['mask'][:, 0].to(device)
                    label_sent, label_sent_mask = input_sent['label'][:, 0].to(device), input_sent['label_mask'][:, 0].to(device)
                    mem_sent, mem_sent_mask = doc_sent['emb'].to(device), doc_sent['mask'].to(device)

                    drop_mask = torch.from_numpy(
                        np.random.binomial(1, 1-config['training']['input_drop'],
                                        (config['max_sent_len'], config['word_edim'])
                                        ).astype(np.float32)).to(device)
                    sent = sent*drop_mask

                    model_output = model(sent, mem_sent, sent_mask=sent_mask, mem_sent_mask=mem_sent_mask)
                    sent_pred = model_output['sents'][0]
                    pred_loss = torch.sum(torch.mean(torch.pow(sent_pred - label_sent, 2), dim=2) * \
                                        (1-label_sent_mask.type(torch.cuda.FloatTensor))) / \
                                torch.sum(1-label_sent_mask.type(torch.cuda.FloatTensor))
                    if doc_sent['idx'] not in memory_sentences_score:
                        memory_sentences_score[doc_sent['idx']] = {'score': float(pred_loss), 'text': doc_sent['text']}
                    else:
                        memory_sentences_score[doc_sent['idx']]['score'] += float(pred_loss)
            pbar.update(1)
        max_score = -1e6
        min_score = 1e6
        for sentence in memory_sentences_score.values():
            if sentence['score'] > max_score:
                max_score = sentence['score']
            if sentence['score'] < min_score:
                min_score = sentence['score']
        for sentence in memory_sentences_score.values():
            norm_score = round(100-(sentence['score']-min_score)/(max_score-min_score)*100)
            print(str(norm_score) + "\t" + sentence['text'])
    exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = SentenceEncoder(config)
# restore_name = 'b24sL32_Adamlr5e-05s10g0.98_memTrue=rnd.cnt1_trs2v0.0.g0.0_maskF0Falsedr0.0.poolmean.sdiffF_v31_trD40_Mha2kMeanPool_2xDns1k_s2vInOutLossNoAttMemory_2sentPred_nDocs_520'
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
    shuffle=True, num_workers=4)
dataset_test = WikiS2vCorrectionBatch(config, valid=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=config['batch_size'],
    shuffle=False, num_workers=0)

# sentence_score_prediction(model, device, dataset_test)

# pretrain
if config['training']['pretrain']:
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    for pretrain_epoch in range(20):
        pretrain(model, device, data_loader_train, optimizer, pretrain_epoch, None)
        dataset_train.on_epoch_end()
    start_epoch += pretrain_epoch

optimizer_warmup = optim.Adam(model.parameters(), lr=1e-5)
optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['lr_step'],
                                                       gamma=config['training']['lr_gamma'])
test_loss = 1e6
for epoch in range(start_epoch, config['training']['epochs'] + start_epoch):
    if epoch < config['training']['warmup']:
        optimizer_ = optimizer_warmup
    else:
        optimizer_ = optimizer
    train(model, device, data_loader_train, optimizer_, epoch, None)

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
