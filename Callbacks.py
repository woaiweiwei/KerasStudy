
import keras
#Sequential顺序模型
from keras.models import Sequential
#导入Dense全连接层与Dropout
from keras.layers import Dense,Dropout
#导入优化器
from keras.optimizers import SGD
#导入callbacks模块
import keras.callbacks as callb
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)



#读取测试数据
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels


#定义每轮结束操作
def epoch_end_operation(epoch):
    #测试模型
    loss,acc = model.evaluate(x_test,y_test,verbose=0)
    print('第%s轮测试后结果：' % epoch)

    print('loss:%.4f' % loss)
    print('############')
    print()
#定义训练结束操作    
def train_end_operation():
    print('GAME OVER!')
    

#创建模型
model = Sequential()

#在模型中添加全连接层
model.add(Dense(256,input_dim=784,activation='tanh'))
model.add(Dense(10,activation='softmax'))

#修改优化器中的学习率
sgd = SGD(lr=0.15)

#编译模型
model.compile(
    #选择自己修改后的优化器
    optimizer=sgd,
    #选择损失函数
    loss='mse',
    #计算精度
    metrics=['accuracy']
)

#相关回调函数
#EarlyStopping
earlystop = callb.EarlyStopping(min_delta=0.0005,patience=4,verbose=1)
#ReduceLROnPlateau
reduceLR = callb.ReduceLROnPlateau(min_delta=0.001,patience=2,verbose=1)


#自定义回调函数
class CaculateLoss(callb.Callback):
    def on_epoch_end(self, epoch, logs={}):
        epoch_end_operation(epoch)
        
class TrainEnd(callb.Callback):
    def on_train_end(self,logs={}):
        train_end_operation()
            
caculate_loss = CaculateLoss()
train_end = TrainEnd()

#需要执行的回调函数列表
callback_list = [earlystop,reduceLR,caculate_loss,train_end]

#训练模型
model.fit(x_train,y_train,validation_split=0.3,batch_size=60,epochs=20,callbacks=callback_list,verbose=0)



