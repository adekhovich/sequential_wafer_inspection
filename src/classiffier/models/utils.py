from .resnet import init_resnet
from .gru import init_gru
from .lstm import init_lstm

custom_models = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet101']


def init_model(model_name, params, pretrained=False):
    if 'resnet' in model_name:
        model = init_resnet(model_name, num_classes=params['num_classes'], 
                            input_channels=params['input_channels'], pretrained=pretrained)
    elif 'gru' in model_name:
        model = init_gru(
            input_dim=params['input_dim'], hidden_dim=params['hidden_dim'], 
            num_classes=params['num_classes'], seq_len=params['seq_len'], 
            num_layers=params['num_layers'], patch_size=params['patch_size'], 
            use_encoder=params['use_encoder'], permute_patches=params['permute_patches']
        )
    elif 'lstm' in model_name:
        model = init_lstm(
            input_dim=params['input_dim'], hidden_dim=params['hidden_dim'], 
            num_classes=params['num_classes'], seq_len=params['seq_len'], 
            num_layers=params['num_layers'], patch_size=params['patch_size'], 
            use_encoder=params['use_encoder'], permute_patches=params['permute_patches']
        )    
            
    return model

        
        
