import tensorflow as tf
import cifar_10_model
import cifar_10_data

_SAVE_PATH="./tensorboard/cifar-10/noreg_withdrop0.6/"

with tf.Session() as sess:
    try:
        print("Trying to restore last checkpoint ...")
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
        cifar_10_model.saver.restore(sess, save_path=last_chk_path)
        print("Restored checkpoint from:", last_chk_path)
    except:
        print("Failed to restore checkpoint. Initializing variables instead.")
        sess.run(tf.global_variables_initializer())
    for epoch in range(cifar_10_model.n_epochs):
        X_batch, y_batch = cifar_10_data.fetch_batch(cifar_10_model.global_step.eval()%cifar_10_data.n_batches,cifar_10_model.batch_size)
        sess.run(cifar_10_model.training_op, feed_dict={cifar_10_model.is_training: True, cifar_10_model.X: X_batch, cifar_10_model.y: y_batch})
        if cifar_10_model.global_step.eval() % 50==0:
            summary_acc_test,acc_test= sess.run([cifar_10_model.Test_acc_summary,cifar_10_model.accuracy],feed_dict={cifar_10_model.is_training:False, cifar_10_model.X: cifar_10_data.test_data[:500],cifar_10_model.y: cifar_10_data.test_labels[:500]})
            summary_acc_train,Loss,acc_train=sess.run([cifar_10_model.Train_acc_summary,cifar_10_model.loss,cifar_10_model.accuracy],feed_dict={cifar_10_model.is_training:False, cifar_10_model.X: X_batch, cifar_10_model.y: y_batch})
            print(cifar_10_model.global_step.eval(), "Train accuracy:", acc_train, "Test accuracy:", acc_test,"Loss:",Loss)
            cifar_10_model.file_writer.add_summary(summary_acc_train,cifar_10_model.global_step.eval())
            cifar_10_model.file_writer.add_summary(summary_acc_test,cifar_10_model.global_step.eval())
            save_path = cifar_10_model.saver.save(sess, save_path=_SAVE_PATH,global_step=cifar_10_model.global_step)

cifar_10_model.file_writer.close()