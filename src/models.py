from agents import *


class Models:

    def __init__(self):
        self.model_type=None

    def get_model(self, model_type, env, learning_rate, model_path=None):

        print_t = False

        if model_type == 'MLP':
            m = 16
            layers = 5
            hidden_size = [m ] *layers
            model = QModelMLP(env.state_shape, env.n_action)
            model.build_model(n_hidden=hidden_size,
                              learning_rate=learning_rate,
                              drop_rate=0.2,
                              activation='tanh')

        elif model_type == 'conv':

            m = 16
            layers = 2
            filter_num = [m ] *layers
            filter_size = [3] * len(filter_num)
            # use_pool = [False, True, False, True]
            # use_pool = [False, False, True, False, False, True]
            use_pool = None
            # dilation = [1,2,4,8]
            dilation = None
            dense_units = [48 ,24]
            model = QModelConv(env.state_shape, env.n_action)
            model.build_model(filter_num, filter_size, dense_units, learning_rate,
                              dilation=dilation, use_pool=use_pool)

        elif model_type == 'RNN':

            m = 32
            layers = 3
            hidden_size = [m ] *layers
            dense_units = [m ,m]
            model = QModelGRU(env.state_shape, env.n_action)
            model.build_model(hidden_size, dense_units, learning_rate=learning_rate)
            print_t = True

        elif model_type == 'ConvRNN':

            m = 8
            conv_n_hidden = [m ,m]
            RNN_n_hidden = [m ,m]
            dense_units = [m ,m]
            model = QModelConvGRU(env.state_shape, env.n_action)
            model.build_model(conv_n_hidden, RNN_n_hidden, dense_units, learning_rate=learning_rate)
            print_t = True

        elif model_type == 'pretrained':
            model = load_model(path, learning_rate)

        else:
            raise ValueError

        return model, print_t
