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

print(int(sys.argv[1]))
config_idx = int(sys.argv[1])
config = configs[int(sys.argv[1])]

if config['training']['log']:
    now = datetime.now()
    writer = SummaryWriter(log_dir="./train/logs/"+config['name'])

def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    train_loss = 0.0

    ord_total = 0.0
    ord_correct = 0.0

    start = time.time()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    criterion_order = torch.nn.CrossEntropyLoss()
    for batch_idx, (sent, sent_mask, mem_sent, mem_sent_mask, label_sent, label_sent_mask, sent_order) in enumerate(train_loader):
        sent, sent_mask = sent.to(device), sent_mask.to(device)
        mem_sent, mem_sent_mask = mem_sent.to(device), mem_sent_mask.to(device)
        label_sent, label_sent_mask = label_sent.to(device), label_sent_mask.to(device)
        sent_order = torch.squeeze(sent_order.to(device))
        optimizer.zero_grad()

        model_output = model(sent, mem_sent, sent_mask=sent_mask, mem_sent_mask=mem_sent_mask)
        mem_s2v = model_output['mem_s2v']

        sent_pred = model_output['sent']
        pred_loss_pos = torch.sum(torch.mean(torch.pow(sent_pred - label_sent, 2), dim=2) * \
                             (1-label_sent_mask.type(torch.cuda.FloatTensor))) / \
                        torch.sum(1-label_sent_mask.type(torch.cuda.FloatTensor))
        pred_loss_neg = torch.sum(torch.mean(torch.pow(sent_pred[:, :-1] - label_sent[:, 1:], 2), dim=2) * \
                             (1-label_sent_mask[:, 1:].type(torch.cuda.FloatTensor))) / \
                        torch.sum(1-label_sent_mask[:, 1:].type(torch.cuda.FloatTensor))
        pred_loss_neg2 = torch.sum(torch.mean(torch.pow(sent_pred[:, :-2] - label_sent[:, 2:], 2), dim=2) * \
                             (1-label_sent_mask[:, 2:].type(torch.cuda.FloatTensor))) / \
                        torch.sum(1-label_sent_mask[:, 2:].type(torch.cuda.FloatTensor))
        pred_loss = pred_loss_pos - torch.nn.functional.threshold(-pred_loss_neg, -0.02, -0.02) \
                                  - torch.nn.functional.threshold(-pred_loss_neg2, -0.02, -0.02)

        sent2_pred = model_output['sent2']
        pred_loss_pos = torch.sum(torch.mean(torch.pow(sent2_pred[:, :-1] - label_sent[:, 1:], 2), dim=2) * \
                             (1-label_sent_mask[:, 1:].type(torch.cuda.FloatTensor))) / \
                        torch.sum(1-label_sent_mask[:, 1:].type(torch.cuda.FloatTensor))
        pred_loss_neg = torch.sum(torch.mean(torch.pow(sent2_pred[:, :-2] - label_sent[:, 2:], 2), dim=2) * \
                             (1-label_sent_mask[:, 2:].type(torch.cuda.FloatTensor))) / \
                        torch.sum(1-label_sent_mask[:, 2:].type(torch.cuda.FloatTensor))
        pred_loss_neg2 = torch.sum(torch.mean(torch.pow(sent2_pred[:, :-3] - label_sent[:, 3:], 2), dim=2) * \
                             (1-label_sent_mask[:, 3:].type(torch.cuda.FloatTensor))) / \
                        torch.sum(1-label_sent_mask[:, 3:].type(torch.cuda.FloatTensor))
        pred_loss += pred_loss_pos - torch.nn.functional.threshold(-pred_loss_neg, -0.02, -0.02) \
                                   - torch.nn.functional.threshold(-pred_loss_neg2, -0.02, -0.02)

        pred_loss_neg_in = torch.sum(torch.mean(torch.pow(sent_pred[:, 1:] - label_sent[:, :-1], 2), dim=2) * \
                                (1-label_sent_mask[:, :-1].type(torch.cuda.FloatTensor))) / \
                           torch.sum(1-label_sent_mask[:, :-1].type(torch.cuda.FloatTensor))
        pred_loss += - torch.nn.functional.threshold(-pred_loss_neg_in, -0.04, -0.04)

        # s2v_in = model_output['s2v'][0]
        # s2v_out = model_output['s2v'][1]
        # s2v_loss = torch.sum(torch.mean(torch.pow(s2v_in - s2v_out, 2), dim=1))
        # pred_loss += s2v_loss

        # if mem_s2v.shape[0] == config['batch_size']:
        #     ord_loss = criterion_order(model_output['order'], sent_order)
        #     pred_loss += ord_loss

        if batch_idx == 0:
            print(mem_s2v)
        train_loss += pred_loss.detach()

        pred_loss.backward(retain_graph=True)
        optimizer.step()

        if scheduler:
            scheduler.step()

        # if mem_s2v.shape[0] == config['batch_size']:
        #     _, pred_idx = torch.max(model_output['order'], 1)
        #     label_idx = sent_order
        #     ord_total += sent_order.size(0)
        #     ord_correct += (pred_idx == label_idx).sum().item()

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
        # writer.add_scalar('ord_acc/train', 100.0*ord_correct/ord_total, epoch)
        writer.flush()

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0.0
    test_s2_loss = 0.0
    test_loss_s2v = 0.0
    # criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (sent, sent_mask, mem_sent, mem_sent_mask, label_sent, label_sent_mask, _) in enumerate(test_loader):
            sent, sent_mask = sent.to(device), sent_mask.to(device)
            mem_sent, mem_sent_mask = mem_sent.to(device), mem_sent_mask.to(device)
            label_sent, label_sent_mask = label_sent.to(device), label_sent_mask.to(device)

            model_output = model(sent, mem_sent, sent_mask=sent_mask, mem_sent_mask=mem_sent_mask)

            sent_pred = model_output['sent']
            pred_loss = torch.sum(torch.mean(torch.pow(sent_pred - label_sent, 2), dim=2) * \
                                 (1-label_sent_mask.type(torch.cuda.FloatTensor))) / \
                        torch.sum(1-label_sent_mask.type(torch.cuda.FloatTensor))
            test_loss += pred_loss.detach()

            sent2_pred = model_output['sent2']
            pred_loss_s2 = torch.sum(torch.mean(torch.pow(sent2_pred[:, :-1] - label_sent[:, 1:], 2), dim=2) * \
                                (1-label_sent_mask[:, 1:].type(torch.cuda.FloatTensor))) / \
                            torch.sum(1-label_sent_mask[:, 1:].type(torch.cuda.FloatTensor))
            test_s2_loss += pred_loss_s2.detach()

            # s2v_in = model_output['s2v'][0]
            # s2v_out = model_output['s2v'][1]
            # s2v_loss = torch.sum(torch.mean(torch.pow(s2v_in - s2v_out, 2), dim=1))
            # test_loss_s2v += s2v_loss.detach()

    test_loss /= batch_idx + 1
    test_s2_loss /= batch_idx + 1
    test_loss_s2v /= batch_idx + 1
    if config['training']['log']:
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.add_scalar('loss/test_s2', test_s2_loss, epoch)
        writer.add_scalar('loss/test_loss_s2v', test_loss_s2v, epoch)
        writer.flush()
    print('\t\tTest set: Average loss: {:.6f}'.format(test_loss))
    return test_loss

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
                    sent_pred = model_output['sent']

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
