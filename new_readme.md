让我帮你分析这个GAN的原始实现代码库的整体架构。

## 主要组件

1. **核心类**
- `AdversaryPair`: GAN的主体类,包含生成器和判别器

```30:101:__init__.py
class AdversaryPair(Model):

    def __init__(self, generator, discriminator, inferer=None,
                 inference_monitoring_batch_size=128,
                 monitor_generator=True,
                 monitor_discriminator=True,
                 monitor_inference=True,
                 shrink_d = 0.):
        Model.__init__(self)
        self.__dict__.update(locals())
        del self.self

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'inferer' not in state:
            self.inferer = None
        if 'inference_monitoring_batch_size' not in state:
            self.inference_monitoring_batch_size = 128  # TODO: HACK
        if 'monitor_generator' not in state:
            self.monitor_generator = True
        if 'monitor_discriminator' not in state:
            self.monitor_discriminator = True
        if 'monitor_inference' not in state:
            self.monitor_inference = True

    def get_params(self):
        p = self.generator.get_params() + self.discriminator.get_params()
        if hasattr(self, 'inferer') and self.inferer is not None:
            p += self.inferer.get_params()
        return p

    def get_input_space(self):
        return self.discriminator.get_input_space()
    def get_weights_topo(self):
        return self.discriminator.get_weights_topo()

    def get_weights(self):
        return self.discriminator.get_weights()

    def get_weights_format(self):
        return self.discriminator.get_weights_format()

    def get_weights_view_shape(self):
        return self.discriminator.get_weights_view_shape()

    def get_monitoring_channels(self, data):
        rval = OrderedDict()

        g_ch = self.generator.get_monitoring_channels(data)
        d_ch = self.discriminator.get_monitoring_channels((data, None))
        samples = self.generator.sample(100)
        d_samp_ch = self.discriminator.get_monitoring_channels((samples, None))

        i_ch = OrderedDict()
        if self.inferer is not None:
            batch_size = self.inference_monitoring_batch_size
            sample, noise, _ = self.generator.sample_and_noise(batch_size)
            i_ch.update(self.inferer.get_monitoring_channels((sample, noise)))

        if self.monitor_generator:
            for key in g_ch:
                rval['gen_' + key] = g_ch[key]
        if self.monitor_discriminator:
            for key in d_ch:
                rval['dis_on_data_' + key] = d_samp_ch[key]
            for key in d_ch:
                rval['dis_on_samp_' + key] = d_ch[key]
        if self.monitor_inference:
            for key in i_ch:
                rval['inf_' + key] = i_ch[key]
        return rval
```


- `Generator`: 生成器类,负责生成样本

```134:233:__init__.py
class Generator(Model):

    def __init__(self, mlp, noise = "gaussian", monitor_ll = False, ll_n_samples = 100, ll_sigma = 0.2):
        Model.__init__(self)
        self.__dict__.update(locals())
        del self.self
        self.theano_rng = MRG_RandomStreams(2014 * 5 + 27)

    def get_input_space(self):
        return self.mlp.get_input_space()

    def sample_and_noise(self, num_samples, default_input_include_prob=1., default_input_scale=1., all_g_layers=False):
        n = self.mlp.get_input_space().get_total_dimension()
        noise = self.get_noise((num_samples, n))
        formatted_noise = VectorSpace(n).format_as(noise, self.mlp.get_input_space())
        if all_g_layers:
            rval = self.mlp.dropout_fprop(formatted_noise, default_input_include_prob=default_input_include_prob, default_input_scale=default_input_scale, return_all=all_g_layers)
            other_layers, rval = rval[:-1], rval[-1]
        else:
            rval = self.mlp.dropout_fprop(formatted_noise, default_input_include_prob=default_input_include_prob, default_input_scale=default_input_scale)
            other_layers = None
        return rval, formatted_noise, other_layers

    def sample(self, num_samples, default_input_include_prob=1., default_input_scale=1.):
        sample, _, _ = self.sample_and_noise(num_samples, default_input_include_prob, default_input_scale)
        return sample
    def inpainting_sample_and_noise(self, X, default_input_include_prob=1., default_input_scale=1.):
        # Very hacky! Specifically for inpainting right half of CIFAR-10 given left half
        # assumes X is b01c
        assert X.ndim == 4
        input_space = self.mlp.get_input_space()
        n = input_space.get_total_dimension()
        image_size = input_space.shape[0]
        half_image = int(image_size / 2)
        data_shape = (X.shape[0], image_size, half_image, input_space.num_channels)

        noise = self.theano_rng.normal(size=data_shape, dtype='float32')
        Xg = T.set_subtensor(X[:,:,half_image:,:], noise)
        sampled_part, noise =  self.mlp.dropout_fprop(Xg, default_input_include_prob=default_input_include_prob, default_input_scale=default_input_scale), noise
        sampled_part = sampled_part.reshape(data_shape)
        rval = T.set_subtensor(X[:, :, half_image:, :], sampled_part)
        return rval, noise


    def get_monitoring_channels(self, data):
        if data is None:
            m = 100
        else:
            m = data.shape[0]
        n = self.mlp.get_input_space().get_total_dimension()
        noise = self.get_noise((m, n))
        rval = OrderedDict()

        try:
            rval.update(self.mlp.get_monitoring_channels((noise, None)))
        except Exception:
            warnings.warn("something went wrong with generator.mlp's monitoring channels")

        if  self.monitor_ll:
            rval['ll'] = T.cast(self.ll(data, self.ll_n_samples, self.ll_sigma),
                                        theano.config.floatX).mean()
            rval['nll'] = -rval['ll']
        return rval
    def get_noise(self, size):

        # Allow just requesting batch size
        if isinstance(size, int):
            size = (size, self.get_input_space().get_total_dimension())

        if not hasattr(self, 'noise'):
            self.noise = "gaussian"
        if self.noise == "uniform":
            return self.theano_rng.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=size, dtype='float32')
        elif self.noise == "gaussian":
            return self.theano_rng.normal(size=size, dtype='float32')
        elif self.noise == "spherical":
            noise = self.theano_rng.normal(size=size, dtype='float32')
            noise = noise / T.maximum(1e-7, T.sqrt(T.sqr(noise).sum(axis=1))).dimshuffle(0, 'x')
            return noise
        else:
            raise NotImplementedError(self.noise)

    def get_params(self):
        return self.mlp.get_params()

    def get_output_space(self):
        return self.mlp.get_output_space()

    def ll(self, data, n_samples, sigma):

        samples = self.sample(n_samples)
        output_space = self.mlp.get_output_space()
        if 'Conv2D' in str(output_space):
            samples = output_space.convert(samples, output_space.axes, ('b', 0, 1, 'c'))
            samples = samples.flatten(2)
            data = output_space.convert(data, output_space.axes, ('b', 0, 1, 'c'))
            data = data.flatten(2)
        parzen = theano_parzen(data, samples, sigma)
```


2. **训练算法**
- 使用修改版的SGD优化器进行训练

```45:167:sgd.py
class SGD(TrainingAlgorithm):
    ...
    def __init__(self, learning_rate, cost=None, batch_size=None,
                 monitoring_batch_size=None, monitoring_batches=None,
                 monitoring_dataset=None, monitor_iteration_mode='sequential',
                 termination_criterion=None, update_callbacks=None,
                 learning_rule = None, init_momentum = None,
                 set_batch_size = False,
                 train_iteration_mode = None, batches_per_iter=None,
                 theano_function_mode = None, monitoring_costs=None,
                 seed=[2012, 10, 5], discriminator_steps=1):
        self.discriminator_steps = discriminator_steps
```


3. **配置文件**
- MNIST训练配置

```1:95:mnist.yaml
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.mnist.MNIST {
        which_set: 'train',
        start: 0,
        stop: 50000
    },
    model: !obj:adversarial.AdversaryPair {
        generator: !obj:adversarial.Generator {
            noise: 'uniform',
            monitor_ll: 1,
            mlp: !obj:pylearn2.models.mlp.MLP {
            layers: [
                     !obj:pylearn2.models.mlp.RectifiedLinear {
                         layer_name: 'h0',
                         dim: 1200,
                         irange: .05,
                     },
                     !obj:pylearn2.models.mlp.RectifiedLinear {
                         layer_name: 'h1',
                         dim: 1200,
                         irange: .05,
                     },
                     !obj:pylearn2.models.mlp.Sigmoid {
                         init_bias: !obj:pylearn2.models.dbm.init_sigmoid_bias_from_marginals { dataset: *train},
                         layer_name: 'y',
                         irange: .05,
                         dim: 784
                     }
                    ],
            nvis: 100,
        }},
        discriminator: 
            !obj:pylearn2.models.mlp.MLP {
            layers: [
                     !obj:pylearn2.models.maxout.Maxout {
                         layer_name: 'h0',
                         num_units: 240,
                         num_pieces: 5,
                         irange: .005,
                     },
                     !obj:pylearn2.models.maxout.Maxout {
                         layer_name: 'h1',
                         num_units: 240,
                         num_pieces: 5,
                         irange: .005,
                     },
                     !obj:pylearn2.models.mlp.Sigmoid {
                         layer_name: 'y',
                         dim: 1,
                         irange: .005
                     }
                    ],
            nvis: 784,
        },
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 100,
        learning_rate: .1,
        learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
            init_momentum: .5,
        },
        monitoring_dataset:
            {
                'valid' : !obj:pylearn2.datasets.mnist.MNIST {
                              which_set: 'train',
                              start: 50000,
                              stop:  60000
                          },
            },
        cost: !obj:adversarial.AdversaryCost2 {
            scale_grads: 0,
            #target_scale: 1.,
            discriminator_default_input_include_prob: .5,
            discriminator_input_include_probs: {
                'h0': .8
            },
            discriminator_default_input_scale: 2.,
            discriminator_input_scales: {
                'h0': 1.25   
            }
            },
        #!obj:pylearn2.costs.mlp.dropout.Dropout {
        #    input_include_probs: { 'h0' : .8 },
        #    input_scales: { 'h0': 1. }
        #},
        #termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
        #    channel_name: "valid_y_misclass",
        #    prop_decrease: 0.,
        #    N: 100
        #},
        update_callbacks: !obj:pylearn2.training_algorithms.sgd.ExponentialDecay {
            decay_factor: 1.000004,
            min_lr: .000001
        }
    },
```


- CIFAR-10训练配置(卷积和全连接两个版本)

```1:174:cifar10_convolutional.yaml
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.cifar10.CIFAR10 {
        axes: ['c', 0, 1, 'b'],
        gcn: 55.,
        which_set: 'train',
        start: 0,
        stop: 40000
    },
    model: !obj:adversarial.AdversaryPair {
        generator: !obj:adversarial.Generator {
            mlp: !obj:pylearn2.models.mlp.MLP {
            layers: [
                     !obj:pylearn2.models.mlp.RectifiedLinear {
                         layer_name: 'gh0',
                         dim: 8000,
                         irange: .05,
                         #max_col_norm: 1.9365,
                     },
                     !obj:pylearn2.models.mlp.Sigmoid {
                         layer_name: 'h1',
                         dim: 8000,
                         irange: .05,
                         #max_col_norm: 1.9365,
                     },
                     !obj:pylearn2.models.mlp.SpaceConverter {
                         layer_name: 'converter',
                         output_space: !obj:pylearn2.space.Conv2DSpace {
                        shape: [10, 10],
                        num_channels: 80,
                        axes: ['c', 0, 1, 'b'],
                    }},
                     !obj:adversarial.deconv.Deconv {
                     #W_lr_scale: .05,
                     #b_lr_scale: .05,
                         num_channels: 3,
                         output_stride: [3, 3],
                         kernel_shape: [5, 5],
                         pad_out: 0,
                         #max_kernel_norm: 1.9365,
                         # init_bias: !obj:pylearn2.models.dbm.init_sigmoid_bias_from_marginals { dataset: *train},
                         layer_name: 'y',
                         irange: .05,
                         tied_b: 0
                     },
                    ],
            nvis: 100,
        }},
        discriminator: 
            !obj:pylearn2.models.mlp.MLP {
            layers: [
                 !obj:pylearn2.models.maxout.MaxoutConvC01B {
                     layer_name: 'dh0',
                     pad: 4,
                     tied_b: 1,
                     #W_lr_scale: .05,
                     #b_lr_scale: .05,
                     num_channels: 32,
                     num_pieces: 2,
                     kernel_shape: [8, 8],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     irange: .005,
                     #max_kernel_norm: .9,
                     partial_sum: 33,
                 },
                 !obj:pylearn2.models.maxout.MaxoutConvC01B {
                     layer_name: 'h1',
                     pad: 3,
                     tied_b: 1,
                     #W_lr_scale: .05,
                     #b_lr_scale: .05,
                     num_channels: 32, # 192 ran out of memory
                     num_pieces: 2,
                     kernel_shape: [8, 8],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     irange: .005,
                     #max_kernel_norm: 1.9365,
                     partial_sum: 15,
                 },
                 !obj:pylearn2.models.maxout.MaxoutConvC01B {
                     pad: 3,
                     layer_name: 'h2',
                     tied_b: 1,
                     #W_lr_scale: .05,
                     #b_lr_scale: .05,
                     num_channels: 192,
                     num_pieces: 2,
                     kernel_shape: [5, 5],
                     pool_shape: [2, 2],
                     pool_stride: [2, 2],
                     irange: .005,
                     #max_kernel_norm: 1.9365,
                 },
                 !obj:pylearn2.models.maxout.Maxout {
                    layer_name: 'h3',
                    irange: .005,
                    num_units: 500,
                    num_pieces: 5,
                    #max_col_norm: 1.9
                     },
                     !obj:pylearn2.models.mlp.Sigmoid {
                         #W_lr_scale: .1,
                         #b_lr_scale: .1,
                         #max_col_norm: 1.9365,
                         layer_name: 'y',
                         dim: 1,
                         irange: .005
                     }
                    ],
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: [32, 32],
            num_channels: 3,
            axes: ['c', 0, 1, 'b'],
        }
        },
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 128,
        learning_rate: .004,
        learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
            init_momentum: .5,
        },
        monitoring_dataset:
            {
                #'train' : *train,
                'valid' : !obj:pylearn2.datasets.cifar10.CIFAR10 {
                            axes: ['c', 0, 1, 'b'],
                              gcn: 55., 
                              which_set: 'train',
                              start: 40000,
                              stop:  50000
                          },
                #'test'  : !obj:pylearn2.datasets.cifar10.CIFAR10 {
                #              which_set: 'test',
                #              gcn: 55.,
                #          }
            },
        cost: !obj:adversarial.AdversaryCost2 {
            scale_grads: 0,
            #target_scale: .1,
            discriminator_default_input_include_prob: .5,
            discriminator_input_include_probs: {
                'dh0': .8
            },
            discriminator_default_input_scale: 2.,
            discriminator_input_scales: {
                'dh0': 1.25   
            }
            },
        #termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
        #    channel_name: "valid_y_misclass",
        #    prop_decrease: 0.,
        #    N: 100
        #},
        update_callbacks: !obj:pylearn2.training_algorithms.sgd.ExponentialDecay {
            decay_factor: 1.000004,
            min_lr: .000001
        }
    },
    extensions: [
        #!obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
        #     channel_name: 'valid_y_misclass',
        #     save_path: "${PYLEARN2_TRAIN_FILE_FULL_STEM}_best.pkl"
        #},
        !obj:pylearn2.training_algorithms.learning_rule.MomentumAdjustor {
            start: 1,
            saturate: 250,
            final_momentum: .7
        }
    ],
    save_path: "${PYLEARN2_TRAIN_FILE_FULL_STEM}.pkl",
    save_freq: 1
}
```


## 主要功能

1. **样本生成与可视化**
- 提供了多个脚本用于生成和展示样本:
  - show_samples_mnist_paper.py
  - show_samples_cifar_conv_paper.py 
  - show_samples_tfd_paper.py

2. **模型评估**
- 使用Parzen窗估计对数似然

```1:124:parzen_ll.py
import argparse
import time
import gc
import numpy
import theano
import theano.tensor as T
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.tfd import TFD



def get_nll(x, parzen, batch_size=10):
    """
    Credit: Yann N. Dauphin
    """

    inds = range(x.shape[0])
    n_batches = int(numpy.ceil(float(len(inds)) / batch_size))

    times = []
    nlls = []
    for i in range(n_batches):
        begin = time.time()
        nll = parzen(x[inds[i::n_batches]])
        end = time.time()
        times.append(end-begin)
        nlls.extend(nll)

        if i % 10 == 0:
            print i, numpy.mean(times), numpy.mean(nlls)

    return numpy.array(nlls)


def log_mean_exp(a):
    """
    Credit: Yann N. Dauphin
    """

    max_ = a.max(1)

    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def theano_parzen(mu, sigma):
    """
    Credit: Yann N. Dauphin
    """

    x = T.matrix()
    mu = theano.shared(mu)
    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma
    E = log_mean_exp(-0.5*(a**2).sum(2))
    Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))

    return theano.function([x], E - Z)
...
def main():
    parser = argparse.ArgumentParser(description = 'Parzen window, log-likelihood estimator')
    parser.add_argument('-p', '--path', help='model path')
    parser.add_argument('-s', '--sigma', default = None)
    parser.add_argument('-d', '--dataset', choices=['mnist', 'tfd'])
    parser.add_argument('-f', '--fold', default = 0, type=int)
    parser.add_argument('-v', '--valid', default = False, action='store_true')
    parser.add_argument('-n', '--num_samples', default=10000, type=int)
    parser.add_argument('-l', '--limit_size', default=1000, type=int)
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    parser.add_argument('-c', '--cross_val', default=10, type=int,
                            help="Number of cross valiation folds")
    parser.add_argument('--sigma_start', default=-1, type=float)
    parser.add_argument('--sigma_end', default=0., type=float)
    args = parser.parse_args()

    # load model
    model = serial.load(args.path)
    src = model.dataset_yaml_src
    batch_size = args.batch_size
    model.set_batch_size(batch_size)

    # load test set
    test = yaml_parse.load(src)
    test = get_test(args.dataset, test, args.fold)

    # generate samples
    samples = model.generator.sample(args.num_samples).eval()
    output_space = model.generator.mlp.get_output_space()
```


## 新的README.md

```markdown
# Generative Adversarial Networks (原始实现)

这是Ian Goodfellow等人2014年GAN论文的原始实现代码。

## 简介

生成对抗网络(GAN)是一个由生成器和判别器组成的深度学习框架。本仓库包含了论文中实验的完整代码实现。

## 主要特性

- 支持MNIST、CIFAR-10和Toronto Face Database(TFD)数据集
- 提供卷积和全连接两种网络架构
- 包含样本生成和可视化工具
- 使用Parzen窗方法进行模型评估

## 依赖

- Theano 
- Pylearn2
- NumPy
- PyYAML

## 使用方法

1. 训练模型:
```bash
pylearn2/scripts/train.py mnist.yaml  # MNIST数据集
pylearn2/scripts/train.py cifar10_convolutional.yaml  # CIFAR-10数据集
```

2. 生成样本:
```bash
python show_samples.py path/to/model.pkl
```

3. 评估模型:
```bash
python parzen_ll.py -p path/to/model.pkl -d mnist
```

## 引用

如果您在研究中使用了本代码,请引用原论文:

```
@article{goodfellow2014generative,
  title={Generative Adversarial Networks},
  author={Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1406.2661},
  year={2014}
}
```

## 许可证

BSD 3-Clause License
```

这个代码库展示了GAN的第一个实现,虽然使用了当时的深度学习框架(Theano),但其核心思想和结构对理解GAN的工作原理非常有帮助。
