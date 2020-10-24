import tensorflow as tf
import copy

default_config = {
    'batch_size': 16,
    'max_sent_len': 64,
    'word_edim': 1024,
    's2v_dim': 2048,
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

# -----------------------------------------------------------------------------
i = 0
for _ in range(0, 4):
    configs[i]['sentence_encoder']['input_drop'] = 0.0
    configs[i]['sentence_encoder']['transformer']['num_layers'] = 4
    configs[i]['sentence_encoder']['transformer']['dropout'] = 0.1
    configs[i]['sentence_encoder']['transformer']['ffn_dim'] = 256
    configs[i]['sentence_encoder']['transformer']['num_heads'] = 16

    configs[i]['sentence_encoder']['pooling']['mha']['inner_dim'] = 2*1024
    configs[i]['sentence_encoder']['pooling']['mha']['num_heads'] = 128
    configs[i]['sentence_encoder']['pooling']['mha']['attention_dropout'] = 0.0

    configs[i]['s2v_dim'] = 2*1024
    # configs[i]['classifier_network']['dropout'] = 0.2
    configs[i]['max_sent_len'] = 32
    configs[i]['batch_size'] = 24

    configs[i]['training']['optimizer'] = 'Adam'
    # configs[i]['training']['label_smoothing'] = 0.2
    # configs[i]['training']['clipnorm'] = 1.0
    configs[i]['training']['lr'] = 8e-5
    configs[i]['training']['lr_step'] = 4
    configs[i]['training']['lr_gamma'] = 0.95

    configs[i]['name'] = 'b' + str(configs[i]['batch_size']) + 'sL' + str(configs[i]['max_sent_len']) + \
        '_' + configs[i]['training']['optimizer'] + 'lr' + str(configs[i]['training']['lr']) + 's' + str(configs[i]['training']['lr_step']) + 'g' + str(configs[i]['trainig']['lr_gamma']) + \
        '_gTr' + str(configs[i]['sentence_encoder']['transformer']['num_layers']) + 'dr' + str(configs[i]['transformer']['dropout']) + \
        '_mha' + str(configs[i]['sentence_encoder']['transformer']['num_heads']) + \
        '_ffn' + str(configs[i]['sentence_encoder']['transformer']['ffn_dim']) + \
        '_pool' + configs[i]['sentence_encoder']['pooling']['pooling_function'] + 'dr' + str(configs[i]['sentence_encoder']['pooling']['mha']['attention_dropout']) + \
        '_s2v' + str(configs[i]['s2v_dim']) + \
        '_' + str(i)
    i += 1
