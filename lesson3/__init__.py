# Lab 2 Linear Regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Try to find values for W and b to compute y_data = x_data * W + b
# We know that W should be 1 and b should be 0
# But let TensorFlow figure it out
# variable이라는 함수가 자동적으로 값을 다시 담이서 W b를 저장함
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = x_train * W + b

# cost/loss function
#  reduce_mean : 평균 값 square: 제곱값
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize

#최소화하는것 그 차를 줄이는 것
# 일단 지금은 매직~
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:# 코스트 값 Wr값 잘 찍히는지 한번 보는 거
        print(step, sess.run(cost), sess.run(W), sess.run(b))

tf.set_random_seed(777)  # for reproducibility


