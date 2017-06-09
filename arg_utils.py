import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def process_args(default_train_vgg=True, default_learning_rate=0.0001, default_dropout_rate=0.5,
                 default_batch_size=60, default_epochs=50, default_smooth=1.0, default_pos_weight=2.0,
                 default_fc1_shape=1024, default_fc2_shape=256):
    ap = argparse.ArgumentParser()

    ap.add_argument("-tv", "--train_vgg", required=False, default=default_train_vgg, dest='train_vgg',
                    type=lambda x: str2bool(x), help="train or not the vgg part of the network")
    ap.add_argument("-lr", "--learning_rate", required=False, default=default_learning_rate,
                    type=lambda x: float(x), help="learning rate")
    ap.add_argument("-dr", "--dropout_rate", required=False, default=default_dropout_rate,
                    type=lambda x: float(x), help="dropout rate")
    ap.add_argument("-bs", "--batch_size", required=False, default=default_batch_size,
                    type=lambda x: int(x), help="batch size")
    ap.add_argument("-e", "--epochs", required=False, default=default_epochs,
                    type=lambda x: int(x), help="epocs count")
    ap.add_argument("-sm", "--smooth", required=False, default=default_smooth,
                    type=lambda x: float(x), help="smoothing")
    ap.add_argument("-pw", "--pow_weight", required=False, default=default_pos_weight,
                    type=lambda x: float(x), help="weight to use on positive examples in loss function")
    ap.add_argument("-fc1", "--fc1", required=False, default=default_fc1_shape,
                    type=lambda x: int(x), help="shape of first fully connected layer")
    ap.add_argument("-fc2", "--fc2", required=False, default=default_fc2_shape,
                    type=lambda x: int(x), help="shape of second fully connected layer")

    args = vars(ap.parse_args())

    return args['train_vgg'], args['learning_rate'], args['dropout_rate'], args['batch_size'],\
           args['epochs'], args['smooth'], args['pow_weight'], args['fc1'], args['fc2']
