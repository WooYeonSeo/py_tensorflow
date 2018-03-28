# Lab 2 Linear Regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# Try to find values for W and b to compute y_data = W * x_data + b
# We know that W should be 1 and b should be 0
# But let's use TensorFlow to figure it out
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Now we can use X and Y in place of x_data and y_data
# # placeholders for a tensor that will be always fed using feed_dict
# See http://stackoverflow.com/questions/36693740/
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Our hypothesis XW+b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train], # 리스트에 넣어서 한번에 실행
                 feed_dict={X: [1, 2, 3], Y: [1, 2, 3]}) # 이쪽에서 바로 값을 넣음 -> 데이터만 따로 줄 수 있음
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# 왜!!
# 값을 따로 넘겨 줄 수 있다. 모델을 만들어 놓고 테스트 부분에서 데이터를 넣으면 됨


print(sess.run(hypothesis, feed_dict={X : [4]}))