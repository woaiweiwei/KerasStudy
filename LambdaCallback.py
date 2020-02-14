
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

#定义每轮操作
def epoch_end_operation():
    #测试模型
    loss,acc = model.evaluate(x_test,y_test,verbose=0)
    print('本轮测试后结果：')

    print('loss:',loss,'accuracy:',acc)
    print('######')
    print()
#定义训练结束后的操作    
def train_end_operation():
    print('GAME OVER!')

#自定义回调函数
#每轮结束后的回调函数
epoch_print_callback = callb.LambdaCallback(
    #定义在每轮结束是操作
    on_epoch_end =lambda epoch,
    #执行操作
    logs: epoch_end_operation()
)
#训练结束后的回调函数
train_end_callback = callb.LambdaCallback(
    on_train_end=lambda 
    logs: train_end_operation()
)

#需要执行的回调函数列表
callback_list = [epoch_print_callback,train_end_callback]

#训练模型
model.fit(x_train,y_train,validation_split=0.3,batch_size=60,epochs=3,callbacks=callback_list,verbose=1)




