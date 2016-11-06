from __future__ import print_function
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

# Network definition
class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
      super(MLP, self).__init__(
            l1 = L.Linear(None, n_units),
            l2 = L.Linear(None, n_out)
            )

    def __call__(self, x):
        return self.l2(F.relu(self.l1(x)))

# Data Preparation
def make_data(N):
    x = np.empty((N,2),dtype=np.float32)
    y = np.empty(N,dtype=np.int32)
    for i in range(N):
        x1 = i%2
        x2 = (i/2)%2
        x[i][0] = x1
        x[i][1] = x2
        y[i] = x1 ** x2
    return chainer.datasets.TupleDataset(x,y)

def main():
    epoch = 20
    batchsize = 100
    unit = 2
    model = L.Classifier(MLP(unit, 2))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    test = make_data(100)
    train = make_data(10000)
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'validation/main/loss',
                'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    # Training
    trainer.run()

    # Results
    x = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
    y = model.predictor(x).data
    for i in range(4):
        print (x[i],np.argmax(y[i]),y[i])

if __name__ == '__main__':
    main()