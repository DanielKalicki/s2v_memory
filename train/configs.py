import tensorflow as tf
import copy

default_config = {
    'batch_size': 16,
    'max_sent_len': 64,
    'word_edim': 1024,
    's2v_dim': 2048,
    'use_memory': True,
    'memory_sentence_pos': '',
    'name': '',
    'restore_name': '',

    'sentence_encoder': {
        'input_drop': 0.0,
        'transformer': {
            'word_dim': 1024,
            'num_layers': 4,
            'num_heads': 16,
            'ffn_dim': 4*1024,
            'dropout': 0.0
        },
        'pooling': {
            'pooling_method': 'mha',
            'mha': {
                'num_heads': 32,
                'dropout': 0.0
            },
            'pooling_activation': None,  # activation function used before pool
            'pooling_function': 'mean',  # ['mean', 'max', 'l2', 'mean_max']
        }
    },

    'sentence_mlm': {
        'input_drop': 0.0,
        'transformer': {
            'word_dim': 1024,
            'num_layers': 4,
            'num_heads': 16,
            'ffn_dim': 4*1024,
            'dropout': 0.0
        }
    },

    'classifier_network': {
        'hidden_dim': 512,
        'in_dropout': 0.0,  # input dropout
        'hidden_dropout': 0.0,  # hidden dropout
        'hidden_activation': 'gelu',
        'num_classes': 3
    },

    'training': {
        'optimizer': 'Adam',
        'clipnorm': 1.,
        'lr': 1e-3,
        'label_smoothing': 0.2,
        'epochs': 1400,
        'log': True
    }
}

configs = []
for i in range(1000):
    configs.append(copy.deepcopy(default_config))

i = 0
# -----------------------------------------------------------------------------
i = 600
for _ in range(0, 100):
    configs[i]['sentence_encoder']['input_drop'] = 0.0
    configs[i]['sentence_encoder']['transformer']['gate'] = False
    configs[i]['sentence_encoder']['transformer']['num_layers'] = 0
    configs[i]['sentence_encoder']['transformer']['dropout'] = 0.00
    configs[i]['sentence_encoder']['transformer']['ffn_dim'] = 256
    configs[i]['sentence_encoder']['transformer']['num_heads'] = 16

    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 2*1024
    configs[i]['sentence_encoder']['pooling']['mha']['num_heads'] = 128
    configs[i]['sentence_encoder']['pooling']['mha']['attention_dropout'] = 0.0
    configs[i]['sentence_encoder']['pooling']['pooling_method'] = 'mha'

    configs[i]['sentence_mlm']['input_drop'] = 0.0
    configs[i]['sentence_mlm']['transformer']['num_layers'] = 4
    configs[i]['sentence_mlm']['transformer']['dropout'] = 0.00
    configs[i]['sentence_mlm']['transformer']['ffn_dim'] = 256
    configs[i]['sentence_mlm']['transformer']['num_heads'] = False
    configs[i]['sentence_mlm']['transformer']['memory_position'] = 'ffn input'
    configs[i]['sentence_mlm']['transformer']['memory_gate'] = False
    configs[i]['sentence_mlm']['transformer']['memory_res_ffn'] = False
    configs[i]['sentence_mlm']['transformer']['mha'] = False
    configs[i]['sentence_mlm']['transformer']['hidden_sentence_drop'] = 0.00
    configs[i]['sentence_mlm']['transformer']['gate'] = False

    configs[i]['s2v_dim'] = 2*1024
    configs[i]['max_sent_len'] = 32
    configs[i]['num_mem_sents'] = 1
    configs[i]['batch_size'] = 24
    configs[i]['training']['set_mask_token_to_0'] = False
    configs[i]['training']['use_mask_token'] = False
    configs[i]['training']['memory_true_s2v_initial_rate'] = 0.0
    configs[i]['training']['memory_true_s2v_gamma'] = 0.0
    configs[i]['training']['memory_sentence_pos'] = "-1"

    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['lr'] = 1e-4
    configs[i]['training']['lr_step'] = 10
    configs[i]['training']['lr_gamma'] = 0.75
    configs[i]['training']['epochs'] = 100
    configs[i]['training']['input_drop'] = 0.0

    configs[i]['training']['sent_diff_loss'] = False

    configs[i]['sentence_mlm']['transformer']['mha'] = False
    configs[i]['training']['num_predictions'] = 1
    configs[i]['training']['warmup'] = 0
    configs[i]['training']['pretrain'] = False

    if i == 600:
        configs[i]['training']['input_drop'] = 0.0
        configs[i]['sentence_encoder']['transformer']['num_layers'] = 0
        configs[i]['sentence_encoder']['transformer']['gate'] = False
        configs[i]['sentence_encoder']['transformer']['dropout'] = 0.00
        configs[i]['sentence_encoder']['pooling']['pooling_function'] = 'mean'
        configs[i]['num_mem_sents'] = 4 # <---
        configs[i]['s2v_dim'] = 1024
        configs[i]['training']['lr'] = 8e-5
        configs[i]['training']['lr_step'] = 10
        configs[i]['training']['lr_gamma'] = 0.97
        configs[i]['training']['epochs'] = 200

    if i == 601:
        configs[i]['training']['input_drop'] = 0.0
        configs[i]['sentence_encoder']['transformer']['num_layers'] = 0
        configs[i]['sentence_encoder']['transformer']['gate'] = False
        configs[i]['sentence_encoder']['transformer']['dropout'] = 0.00
        configs[i]['sentence_encoder']['pooling']['pooling_function'] = 'mean'
        configs[i]['num_mem_sents'] = 2 # <---
        configs[i]['s2v_dim'] = 1024
        configs[i]['training']['lr'] = 8e-5
        configs[i]['training']['lr_step'] = 10
        configs[i]['training']['lr_gamma'] = 0.97
        configs[i]['training']['epochs'] = 200

    if i == 602:
        configs[i]['training']['input_drop'] = 0.0
        configs[i]['sentence_encoder']['transformer']['num_layers'] = 0
        configs[i]['sentence_encoder']['transformer']['gate'] = False
        configs[i]['sentence_encoder']['transformer']['dropout'] = 0.00
        configs[i]['sentence_encoder']['pooling']['pooling_function'] = 'mean'
        configs[i]['num_mem_sents'] = 1 # <---
        configs[i]['s2v_dim'] = 1024
        configs[i]['training']['lr'] = 8e-5
        configs[i]['training']['lr_step'] = 10
        configs[i]['training']['lr_gamma'] = 0.97
        configs[i]['training']['epochs'] = 200

    if i == 603:
        configs[i]['training']['input_drop'] = 0.0
        configs[i]['sentence_encoder']['transformer']['num_layers'] = 1 # <--
        configs[i]['sentence_encoder']['transformer']['gate'] = False
        configs[i]['sentence_encoder']['transformer']['dropout'] = 0.00
        configs[i]['sentence_encoder']['pooling']['pooling_function'] = 'mean'
        configs[i]['num_mem_sents'] = 1
        configs[i]['s2v_dim'] = 1024
        configs[i]['training']['lr'] = 8e-5
        configs[i]['training']['lr_step'] = 10
        configs[i]['training']['lr_gamma'] = 0.97
        configs[i]['training']['epochs'] = 200

    if i == 604:
        configs[i]['training']['input_drop'] = 0.0
        configs[i]['sentence_encoder']['transformer']['num_layers'] = 0
        configs[i]['sentence_encoder']['transformer']['gate'] = False
        configs[i]['sentence_encoder']['transformer']['dropout'] = 0.00
        configs[i]['sentence_encoder']['pooling']['pooling_function'] = 'mean'
        configs[i]['num_mem_sents'] = 1
        configs[i]['s2v_dim'] = 2048 # <----
        configs[i]['training']['lr'] = 8e-5
        configs[i]['training']['lr_step'] = 10
        configs[i]['training']['lr_gamma'] = 0.97
        configs[i]['training']['epochs'] = 200

    if i == 605:
        configs[i]['training']['input_drop'] = 0.0
        configs[i]['sentence_encoder']['transformer']['num_layers'] = 0
        configs[i]['sentence_encoder']['transformer']['gate'] = False
        configs[i]['sentence_encoder']['transformer']['dropout'] = 0.00
        configs[i]['sentence_encoder']['pooling']['pooling_function'] = 'mean'
        configs[i]['num_mem_sents'] = 1
        configs[i]['s2v_dim'] = 1024
        configs[i]['training']['lr'] = 8e-5
        configs[i]['training']['lr_step'] = 10
        configs[i]['training']['lr_gamma'] = 0.97
        configs[i]['training']['epochs'] = 200
        configs[i]['training']['num_predictions'] = 2 # <----

    if i == 610:
        configs[i]['training']['input_drop'] = 0.0
        configs[i]['sentence_encoder']['transformer']['num_layers'] = 0
        configs[i]['sentence_encoder']['transformer']['gate'] = False
        configs[i]['sentence_encoder']['transformer']['dropout'] = 0.00
        configs[i]['sentence_encoder']['pooling']['pooling_function'] = 'mean'
        configs[i]['num_mem_sents'] = 1
        configs[i]['s2v_dim'] = 1024
        configs[i]['training']['lr'] = 8e-5
        configs[i]['training']['lr_step'] = 10
        configs[i]['training']['lr_gamma'] = 0.97
        configs[i]['training']['epochs'] = 2000
        configs[i]['training']['num_predictions'] = 1
        # configs[i]['sentence_mlm']['transformer']['num_heads'] = 16
        # configs[i]['sentence_mlm']['transformer']['mha'] = True

    if i == 611:
        configs[i]['training']['input_drop'] = 0.0
        configs[i]['sentence_encoder']['transformer']['num_layers'] = 0
        configs[i]['sentence_encoder']['transformer']['gate'] = False
        configs[i]['sentence_encoder']['transformer']['dropout'] = 0.00
        configs[i]['sentence_encoder']['pooling']['pooling_function'] = 'mean'
        configs[i]['num_mem_sents'] = 1
        configs[i]['s2v_dim'] = 1024
        configs[i]['training']['lr'] = 8e-5
        configs[i]['training']['lr_step'] = 10
        configs[i]['training']['lr_gamma'] = 0.97
        configs[i]['training']['epochs'] = 2000
        configs[i]['training']['num_predictions'] = 1
        configs[i]['sentence_mlm']['transformer']['num_heads'] = 16
        configs[i]['sentence_mlm']['transformer']['mha'] = True

    if i == 612:
        configs[i]['training']['input_drop'] = 0.0
        configs[i]['sentence_encoder']['transformer']['num_layers'] = 0
        configs[i]['sentence_encoder']['transformer']['gate'] = False
        configs[i]['sentence_encoder']['transformer']['dropout'] = 0.00
        configs[i]['sentence_encoder']['pooling']['pooling_function'] = 'mean'
        configs[i]['num_mem_sents'] = 1
        configs[i]['s2v_dim'] = 1024
        configs[i]['training']['lr'] = 8e-5
        configs[i]['training']['lr_step'] = 10
        configs[i]['training']['lr_gamma'] = 0.97
        configs[i]['training']['epochs'] = 2000
        configs[i]['training']['num_predictions'] = 1
        configs[i]['sentence_mlm']['transformer']['num_heads'] = 16
        configs[i]['sentence_mlm']['transformer']['mha'] = True

    if i == 613:
        configs[i]['training']['input_drop'] = 0.0
        configs[i]['sentence_encoder']['transformer']['num_layers'] = 0
        configs[i]['sentence_encoder']['transformer']['gate'] = False
        configs[i]['sentence_encoder']['transformer']['dropout'] = 0.00
        configs[i]['sentence_encoder']['pooling']['pooling_function'] = 'mean'
        configs[i]['num_mem_sents'] = 1
        configs[i]['s2v_dim'] = 1024
        configs[i]['training']['lr'] = 8e-5
        configs[i]['training']['lr_step'] = 10
        configs[i]['training']['lr_gamma'] = 0.97
        configs[i]['training']['epochs'] = 2000
        configs[i]['training']['num_predictions'] = 1
        # configs[i]['sentence_mlm']['transformer']['num_heads'] = 16
        # configs[i]['sentence_mlm']['transformer']['mha'] = True

    if i == 615:
        configs[i]['training']['input_drop'] = 0.0
        configs[i]['sentence_encoder']['transformer']['num_layers'] = 0
        configs[i]['sentence_encoder']['transformer']['gate'] = False
        configs[i]['sentence_encoder']['transformer']['dropout'] = 0.00
        configs[i]['sentence_encoder']['pooling']['pooling_function'] = 'mean'
        configs[i]['num_mem_sents'] = 1
        configs[i]['s2v_dim'] = 1024
        configs[i]['training']['lr'] = 8e-5
        configs[i]['training']['lr_step'] = 10
        configs[i]['training']['lr_gamma'] = 0.97
        configs[i]['training']['epochs'] = 2000
        configs[i]['training']['num_predictions'] = 1
        configs[i]['max_sent_len'] = 48

    if i == 616:
        configs[i]['training']['input_drop'] = 0.0
        configs[i]['sentence_encoder']['transformer']['num_layers'] = 0
        configs[i]['sentence_encoder']['transformer']['gate'] = False
        configs[i]['sentence_encoder']['transformer']['dropout'] = 0.00
        configs[i]['sentence_encoder']['pooling']['pooling_function'] = 'mean'
        configs[i]['num_mem_sents'] = 1
        configs[i]['s2v_dim'] = 1024
        configs[i]['training']['lr'] = 8e-5
        configs[i]['training']['lr_step'] = 10
        configs[i]['training']['lr_gamma'] = 0.97
        configs[i]['training']['epochs'] = 2000
        configs[i]['training']['num_predictions'] = 1
        configs[i]['max_sent_len'] = 48
        configs[i]['sentence_encoder']['input_drop'] = 0.0
        configs[i]['sentence_mlm']['transformer']['num_heads'] = 16
        configs[i]['sentence_mlm']['transformer']['mha'] = True
        configs[i]['sentence_mlm']['transformer']['memory_position'] = 'ffn input'
        configs[i]['sentence_mlm']['transformer']['ffn_dim'] = 2048
        configs[i]['sentence_mlm']['transformer']['num_layers'] = 4
        configs[i]['sentence_mlm']['transformer']['gate'] = True

    mem_pos = ""
    for mp in configs[i]['sentence_mlm']['transformer']['memory_position'].split(', '):
        try:
            mem_pos += mp.split(" ")[0][0] + mp.split(" ")[1][0]
        except:
            pass

    configs[i]['name'] = 'b' + str(configs[i]['batch_size']) + 'sL' + str(configs[i]['max_sent_len']) + \
        '_' + configs[i]['training']['optimizer'] + 'lr' + str(configs[i]['training']['lr']) + \
        's' + str(configs[i]['training']['lr_step']) + 'g' + str(configs[i]['training']['lr_gamma']) + \
        '_mem' + str(configs[i]['use_memory']) + '=' + configs[i]['training']['memory_sentence_pos'] + \
        '.cnt' + str(configs[i]['num_mem_sents']) + \
        '.g' + str(configs[i]['training']['memory_true_s2v_gamma']) + \
        'dr' + str(configs[i]['training']['input_drop']) + \
        '.pool' + configs[i]['sentence_encoder']['pooling']['pooling_function'] + \
        '.s2v' + str(configs[i]['s2v_dim']) + \
        '_mTr' + str(configs[i]['sentence_mlm']['transformer']['num_layers']) + \
        'idr' + str(configs[i]['sentence_mlm']['input_drop']) + \
        '.mha' + str(configs[i]['sentence_mlm']['transformer']['num_heads']) + \
        '.ffn' + str(configs[i]['sentence_mlm']['transformer']['ffn_dim']) + \
        '.hdr' + str(configs[i]['sentence_mlm']['transformer']['hidden_sentence_drop']) + \
        '.g' + str(configs[i]['sentence_mlm']['transformer']['gate'])[0] + \
        '.' + mem_pos + \
        '.' + configs[i]['sentence_encoder']['pooling']['pooling_method'] + \
        '.nPred' + str(configs[i]['training']['num_predictions']) + \
        '_v92_inLLoutLL_sent+-3_s2vGTrx0.mhaPool.nGate.nNorm_mTrHhaH1k_trD40_maskedSentLLayDr.3othDocFix.1Zero_rndMem_lossFullSent_' + str(i)
    i += 1

        # '_v89_sent+-3_s2vGTrx0.mhaPool.nGate.nNorm_trD40_memGateFfn_2xDns1kConv3_n3WmeanLlayer_in.3Mask0_' + str(i)
        # '_v83_sent+-3_s2vGTrx0.mhaPool.nGate.nNorm_trD80_memGateFfn_2xDns1024_3xConv_crossEntr2xFc(4x)_noisePred_' + str(i)
        # '_v72_NegInSentLoss_sent+-3_s2vGTrx0.mhaPool.nGate.nNorm_trD80_memGateFfn_2xDns1024_' + str(i)

        # '_mTr' + str(configs[i]['sentence_mlm']['transformer']['num_layers']) + \
        # 'idr' + str(configs[i]['sentence_mlm']['input_drop']) + \
        # '.mha' + str(configs[i]['sentence_mlm']['transformer']['num_heads']) + \
        # '.ffn' + str(configs[i]['sentence_mlm']['transformer']['ffn_dim']) + \
        # '.hdr' + str(configs[i]['sentence_mlm']['transformer']['hidden_sentence_drop']) + \

        # '_v31_nW_nloss_trD40_s2v2xDns1k_3xDns4k_s2vInOutLossNoAtt1e4_nDocs_' + str(i)
        # '_v19_memGffn+ffn_memCosLos8e-5.8_normloss_trDoc40_' + str(i)
        # '_v28_sOrd_1wmask_sdifOthDocGrDoc.1_nloss_trD40_' + str(i)

        # '_mTr' + str(configs[i]['sentence_mlm']['transformer']['num_layers']) + \
        # 'idr' + str(configs[i]['sentence_mlm']['input_drop']) + \
        # '.mha' + str(configs[i]['sentence_mlm']['transformer']['num_heads']) + \
        # '.ffn' + str(configs[i]['sentence_mlm']['transformer']['ffn_dim']) + \
        # '.hdr' + str(configs[i]['sentence_mlm']['transformer']['hidden_sentence_drop']) + \

        # '_gTr' + str(configs[i]['sentence_encoder']['transformer']['num_layers']) + \
        # '.dr' + str(configs[i]['sentence_encoder']['transformer']['dropout']) + \
        # '.mha' + str(configs[i]['sentence_encoder']['transformer']['num_heads']) + \
        # '.ffn' + str(configs[i]['sentence_encoder']['transformer']['ffn_dim']) + \
        # '.gate' + str(configs[i]['sentence_encoder']['transformer']['gate'])[0] + \
        # '.pool' + configs[i]['sentence_encoder']['pooling']['pooling_function'] + \
        # '' + configs[i]['sentence_encoder']['pooling']['pooling_method'] + \
        # '.s2v' + str(configs[i]['s2v_dim']) + \
        # '_mTr' + str(configs[i]['sentence_mlm']['transformer']['num_layers']) + \
        # 'idr' + str(configs[i]['sentence_mlm']['input_drop']) + \
        # '.mha' + str(configs[i]['sentence_mlm']['transformer']['num_heads']) + \
        # '.ffn' + str(configs[i]['sentence_mlm']['transformer']['ffn_dim']) + \
        # '.hdr' + str(configs[i]['sentence_mlm']['transformer']['hidden_sentence_drop']) + \
        # '.poNorm.' + mem_pos + \
        # '.mGateTanh' + str(configs[i]['sentence_mlm']['transformer']['memory_gate'])[0] + \
        # '.mRes' + str(configs[i]['sentence_mlm']['transformer']['memory_res_ffn'])[0] + \
        # '.gate' + str(configs[i]['sentence_mlm']['transformer']['gate'])[0] + \