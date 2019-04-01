import tensorflow as tf
import datasets.casia_webface as webface
import models.inception_resnet_v1 as facenet

BATCH_SIZE = 32


def triple_loss(y_true, y_pred, batch_size=BATCH_SIZE, alpha=0.2):
    print('True Shape:' + str(y_true.get_shape()))
    print('Pred Shape:' + str(y_pred.get_shape()))

    anchor = y_pred[0:batch_size, :]
    positive = y_pred[batch_size:batch_size+batch_size, :]
    negative = y_pred[batch_size+batch_size:batch_size*3, :]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss


def train_from_folder():
    tf.keras.backend.set_learning_phase(1)  # 设置成训练模式（默认）
    data_gen = webface.CASIAWebFaceSequence('/datasets/CASIA-WebFace_aligned', target_shape=[149, 149], batch_size=BATCH_SIZE, shuffle=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/facenet_rgb-{epoch:02d}.h5'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    model = facenet.InceptionResnetV1()
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=triple_loss,
                  metrics=[tf.keras.metrics.categorical_accuracy])
    model.fit_generator(data_gen, epochs=20, max_queue_size=10, workers=8, callbacks=callbacks)
    model.save_weights('./weights/facenet_rgb_weights.h5')
    model.save('./facenet_rgb.h5')


if __name__ == '__main__':
    # TODO
