import keras
#Sequential顺序模型
from keras.models import Sequential
#导入Dense全连接层与Dropout
from keras.layers import Dense,Dropout
#导入优化器
from keras.optimizers import SGD
from keras.models import load_model
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)



#读取训练数据
x_train = mnist.train.images
y_train = mnist.train.labels

#创建模型
model = Sequential()

#在模型中添加全连接层
model.add(Dense(256,input_dim=784,activation='tanh'))
model.add(Dense(10,activation='softmax'))

#修改优化器中的学习率
sgd = SGD(lr=0.15,decay=0.001)

#编译模型
model.compile(
    #选择自己修改后的优化器
    optimizer=sgd,
    #选择损失函数
    loss='mse',
    #计算精度
    metrics=['accuracy']
)

#训练模型
model.fit(x_train,y_train,batch_size=60,epochs=4)

#保存模型,这里是相对路劲
model.save('save_model.h5')


#读取测试数据
x_test = mnist.test.images
y_test = mnist.test.labels

#载入模型
model = load_model('save_model.h5')

#测试模型
loss,acc = model.evaluate(x_test,y_test)

print('loss:',loss)
print('accuracy:',acc)

#训练模型
model.fit(x_train,y_train,batch_size=60,epochs=3)

#再次训练后保存模型,这里是相对路劲
model.save('save_model1.h5')

