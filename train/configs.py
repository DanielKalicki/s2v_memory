import tensorflow as tf
import copy

default_config = {
    'batch_size': 16,
    'max_sent_len': 64,
    'word_edim': 1024,
    's2v_dim': 2048,
    'use_memory': True,
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
        'optimizer': 'Nadam',
        'clipnorm': 1.,
        'lr': 1e-3,
        'label_smoothing': 0.2,
        'epochs': 1400,
        'log': True
    }
}

configs = []
for i in range(100):
    configs.append(copy.deepcopy(default_config))

i = 0
# -----------------------------------------------------------------------------
# i = 0
for _ in range(0, 5):
    configs[i]['sentence_encoder']['input_drop'] = 0.0
    configs[i]['sentence_encoder']['transformer']['num_layers'] = 0
    configs[i]['sentence_encoder']['transformer']['dropout'] = 0.0
    configs[i]['sentence_encoder']['transformer']['ffn_dim'] = 256
    configs[i]['sentence_encoder']['transformer']['num_heads'] = 16

    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 2*1024
    configs[i]['sentence_encoder']['pooling']['mha']['num_heads'] = 128
    configs[i]['sentence_encoder']['pooling']['mha']['attention_dropout'] = 0.0

    configs[i]['sentence_mlm']['input_drop'] = 0.0
    configs[i]['sentence_mlm']['transformer']['num_layers'] = 4
    configs[i]['sentence_mlm']['transformer']['dropout'] = 0.0
    configs[i]['sentence_mlm']['transformer']['ffn_dim'] = 256
    configs[i]['sentence_mlm']['transformer']['num_heads'] = 16
    configs[i]['sentence_mlm']['transformer']['memory_position'] = 'ffn input, ffn hidden, mha hidden'
    # if i == 0:
    #     configs[i]['sentence_mlm']['transformer']['memory_position'] = 'ffn input, ffn hidden, mha hidden'
    # elif i == 1:
    #     configs[i]['sentence_mlm']['transformer']['memory_position'] = 'ffn input'
    # elif i == 2:
    #     configs[i]['sentence_mlm']['transformer']['memory_position'] = 'ffn hidden'
    # elif i == 3:
    #     configs[i]['sentence_mlm']['transformer']['memory_position'] = 'mha hidden'
    # elif i == 4:
    #     configs[i]['sentence_mlm']['transformer']['memory_position'] = 'ffn input, ffn hidden'

    configs[i]['s2v_dim'] = 2*1024
    configs[i]['max_sent_len'] = 32
    configs[i]['batch_size'] = 24
    if i == 0:
        configs[i]['use_memory'] = True
        configs[i]['memory_sentence_pos'] = "+1"
    elif i == 1:
        configs[i]['use_memory'] = True
        configs[i]['memory_sentence_pos'] = "-1"
    elif i == 2:
        configs[i]['use_memory'] = True
        configs[i]['memory_sentence_pos'] = "rnd"
    elif i == 3:
        configs[i]['use_memory'] = True
        configs[i]['memory_sentence_pos'] = "sent0"
    elif i == 4:
        configs[i]['use_memory'] = False
        configs[i]['memory_sentence_pos'] = ""
    elif i == 4:
        configs[i]['use_memory'] = True
        configs[i]['memory_sentence_pos'] = "maskSent"

    configs[i]['training']['optimizer'] = 'Adam'
    configs[i]['training']['lr'] = 4e-4
    configs[i]['training']['lr_step'] = 10
    configs[i]['training']['lr_gamma'] = 0.5
    configs[i]['training']['epochs'] = 100

    mem_pos = ""
    for mp in configs[i]['sentence_mlm']['transformer']['memory_position'].split(', '):
        try:
            mem_pos += mp.split(" ")[0][0] + mp.split(" ")[1][0]
        except:
            pass

    configs[i]['name'] = 'b' + str(configs[i]['batch_size']) + 'sL' + str(configs[i]['max_sent_len']) + \
        '_' + configs[i]['training']['optimizer'] + 'lr' + str(configs[i]['training']['lr']) + 's' + str(configs[i]['training']['lr_step']) + 'g' + str(configs[i]['training']['lr_gamma']) + \
        '_gTr' + str(configs[i]['sentence_encoder']['transformer']['num_layers']) + \
        '.mha' + str(configs[i]['sentence_encoder']['transformer']['num_heads']) + \
        '.ffn' + str(configs[i]['sentence_encoder']['transformer']['ffn_dim']) + \
        '.pool' + configs[i]['sentence_encoder']['pooling']['pooling_function'] + \
        '.s2v' + str(configs[i]['s2v_dim']) + \
        '.noGate' + \
        '_mTr' + str(configs[i]['sentence_mlm']['transformer']['num_layers']) + 'idr' + str(configs[i]['sentence_mlm']['input_drop']) + \
        '.mha' + str(configs[i]['sentence_mlm']['transformer']['num_heads']) + \
        '.ffn' + str(configs[i]['sentence_mlm']['transformer']['ffn_dim']) + \
        '.postNorm.' + mem_pos + \
        '.memGateTanh' + \
        '_mem' + str(configs[i]['use_memory']) + '=' + configs[i]['memory_sentence_pos'] + \
        '_inMaskW0_v7_normLoss_trDoc10_' + str(i)
    i += 1
        # '_inMaskW0_v5_normLoss_10%tr=sent.epoch40' + str(i)