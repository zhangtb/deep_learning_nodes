import tensorflow as tf

Scalar = tf.constant([18])
Vector = tf.constant([1, 2, 3, 4, 5, 6])
Matrix = tf.constant([[1, 5, 8], [3, 5, 9], [11, 25, 67]])
Tensor = tf.constant([[[1, 5, 6], [2, 6, 9], [29, 4, 6]], [[2, 6, 3], [3, 19, 6], [23, 26, 28]],
                      [[100, 23, 21], [123, 232, 22], [12, 34, 22]]])

print tf.shape(Scalar)
print tf.shape(Vector)
print tf.shape(Matrix)
print tf.shape(Tensor)

print "\n-----------------------------\n"

print tf.rank(Scalar)
print tf.rank(Vector)
print tf.rank(Matrix)
print tf.rank(Tensor)
print "\n-----------------------------\n"




with tf.Session() as session:
    result = session.run(Scalar)
    print result

    result = session.run(Vector)
    print result

    result = session.run(Matrix)
    print result

    result = session.run(Tensor)
    print result
