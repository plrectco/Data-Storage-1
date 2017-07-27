# Multi-class (Nonlinear) SVM Example
#----------------------------------
#
# This function wll illustrate how to
# implement the gaussian kernel with
# multiple classes on the iris dataset.
#
# Gaussian Kernel:
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
#
# X : (Sepal Length, Petal Width)
# Y: (I. setosa, I. virginica, I. versicolor) (3 classes)
#
# Basic idea: introduce an extra dimension to do
# one vs all classification.
#
# The prediction of a point will be the category with
# the largest margin or distance to boundary.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
import MySQLdb
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# read 2 columns as feature, 2d is easier for visualizing
# read the fifth column
# only two classes classification
# localfn='iris.csv'
# x_vals = np.genfromtxt(localfn,delimiter=',',usecols=(0,3)).astype(np.float32) 
# y_iris = np.genfromtxt(localfn,delimiter=',',usecols=(4),dtype=str)

# read from database
xsql="select petal_length, sepal_width from flower"
ysql = "select category from flower"
conn= MySQLdb.connect(
        host='localhost',
        port = 3306,
        user='root',
        passwd='123',
        db ='iris',
        )
cur = conn.cursor()
try:
    xfetch = cur.fetchmany(cur.execute(xsql))
    yfetch = cur.fetchmany(cur.execute(ysql))
    x_vals = np.array(xfetch).astype(np.float32)
    y_iris = np.array(yfetch)
    conn.commit()
except:
    conn.rollback()
    print("Error when reading from the database")
cur.close()
conn.close()



y_vals1 = np.array([1 if y=="Iris-setosa" else -1 for y in y_iris])
y_vals2 = np.array([1 if y=="Iris-versicolor" else -1 for y in  y_iris])
y_vals3 = np.array([1 if y=="Iris-virginica" else -1 for y in  y_iris])
y_vals = np.array([y_vals1, y_vals2, y_vals3])
class1_x = [x[0] for i,x in enumerate(x_vals) if y_iris[i]=="Iris-setosa"]
class1_y = [x[1] for i,x in enumerate(x_vals) if y_iris[i]=="Iris-setosa"]
class2_x = [x[0] for i,x in enumerate(x_vals) if y_iris[i]=="Iris-versicolor"]
class2_y = [x[1] for i,x in enumerate(x_vals) if y_iris[i]=="Iris-versicolor"]
class3_x = [x[0] for i,x in enumerate(x_vals) if y_iris[i]=="Iris-virginica"]
class3_y = [x[1] for i,x in enumerate(x_vals) if y_iris[i]=="Iris-virginica"]

# Declare batch size
batch_size = 50

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32,name="input_x")
y_target = tf.placeholder(shape=[3, None], dtype=tf.float32,name="input_y")
prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32,name="prediction")

# Create variables for svm
b = tf.Variable(tf.random_normal(shape=[3,batch_size]),name="lagrangian")

# Gaussian (RBF) kernel
gamma = tf.constant(-10.0)
# dist = tf.reduce_sum(tf.square(x_data), 1)
# dist = tf.reshape(dist, [-1,1])
# sq_dists = tf.add(tf.sub(dist, tf.mul(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
# my_kernel = tf.exp(tf.mul(gamma, tf.abs(sq_dists)))

# Declare function to do reshape/batch multiplication
def reshape_matmul(mat): 
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [3, batch_size, 1])
    return(tf.batch_matmul(v2, v1)) #[..., r_x, c_x] and [..., r_y, c_y] -> [...,r_x,c_y]


with tf.name_scope("Kernel"):
    # Gaussian (RBF) prediction kernel
    rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
    rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
    pred_sq_dist = tf.add(tf.sub(rA, tf.mul(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
    pred_kernel = tf.exp(tf.mul(gamma, tf.abs(pred_sq_dist)))
    
with tf.name_scope("Model"):
    # Compute SVM Model
    first_term = tf.reduce_sum(b)
    b_vec_cross = tf.matmul(tf.transpose(b), b)
    y_target_cross = reshape_matmul(y_target)
    second_term = tf.reduce_sum(tf.mul(pred_kernel, tf.mul(b_vec_cross, y_target_cross)),[1,2])
    loss = tf.reduce_sum(tf.neg(tf.sub(first_term, second_term)))

    prediction_output = tf.matmul(tf.mul(y_target,b), pred_kernel)
    prediction = tf.arg_max(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1), 0)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target,0)), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.initialize_all_variables()
sess.run(init)

# Initiaize summary
logs_path = "./log"
tf.scalar_summary("loss", loss)
tf.scalar_summary("accuracy", accuracy)
summary_op = tf.merge_all_summaries()
writer = tf.train.SummaryWriter(logs_path,graph=tf.get_default_graph())

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Training loop
loss_vec = []
batch_accuracy = []
for i in range(500):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = y_vals[:,rand_index]
    # sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    _,summary=sess.run([train_step,summary_op], feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid:rand_x})
    writer.add_summary(summary, i)
    # temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    # loss_vec.append(temp_loss)

    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = y_vals[:,rand_index]
    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
                                              y_target: rand_y,
                                              prediction_grid:rand_x})
    # batch_accuracy.append(acc_temp)
    
    if (i+1)%20==0:
    #  Restore the variables  
        # saver.restore(sess, "/tmp/model50.ckpt")
    # Save the variables to disk.
        # save_path = saver.save(sess, "/tmp/model%d.ckpt"%(i+1))
        # print(sess.run(b))
        # print("Model saved in file: %s" % save_path)

        print('Step #' + str(i+1))
        print('Accuracy = ' + str(acc_temp))

# Create a mesh to plot points in
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = sess.run(prediction, feed_dict={x_data: rand_x,
                                                   y_target: rand_y,
                                                   prediction_grid: grid_points})
grid_predictions = grid_predictions.reshape(xx.shape)


# Plot points and grid
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='I. setosa')
plt.plot(class2_x, class2_y, 'kx', label='I. versicolor')
plt.plot(class3_x, class3_y, 'gv', label='I. virginica')
plt.title('Gaussian SVM Results on Iris Data')
plt.xlabel('Pedal Length')
plt.ylabel('Sepal Width')
plt.legend(loc='lower right')
#plt.ylim([-0.5, 3.0])
#plt.xlim([3.5, 8.5])
plt.show()

# # Plot batch accuracy
# plt.plot(batch_accuracy, 'k-', label='Accuracy')
# plt.title('Batch Accuracy')
# plt.xlabel('Generation')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()

# # Plot loss over time
# plt.plot(loss_vec, 'k-')
# plt.title('Loss per Generation')
# plt.xlabel('Generation')
# plt.ylabel('Loss')
# plt.show()