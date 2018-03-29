# Lab 4 Multi-variable linear regression
# 파일로 읽어오기!!!

# reader = tf.TextLineReader(skip_header_lines=1)
# csv의 헤더 type 오류가 나면 skip_header_lines로 skip
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

#numpy 사용 - 데이터 처리에 유용한 라이브러리로 알고있음
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32) # ,로 나누고, 파일의 데이터타입도 준다(일단 지금은 전체의 데이터가 같은 데이터타입)
# 파일에서 슬라이싱해서 x와 y데이터를 나눔
x_data = xy[:, 0:-1] # -1 은 끝값 빼고 가지고오는거
y_data = xy[:, [-1]] # 이게 마지막값

# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias') # rufrhk

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

# Ask my score
print("Your score will be ", sess.run(
    hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ", sess.run(hypothesis,
                                        feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))