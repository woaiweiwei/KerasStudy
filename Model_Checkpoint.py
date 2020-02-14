
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

#保存地址
save_path = 'checkpoint/model-{epoch:02d}.h5'

#相关回调函数
#ModelCheckpoint
checkpoint = callb.ModelCheckpoint(filepath=save_path,verbose=1,period=2)

#需要执行的回调函数列表
callback_list = [checkpoint]

#训练模型
model.fit(x_train,y_train,batch_size=60,epochs=6,callbacks=callback_list,verbose=0)




