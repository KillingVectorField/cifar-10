import tensorflow as tf
import numpy as np
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
import sklearn
from matplotlib import pyplot as plt
import cifar_10_data
import cifar_10_model

with tf.Session() as sess:
    try:
        print("Trying to restore last checkpoint ...")
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=cifar_10_model._SAVE_PATH)
        cifar_10_model.saver.restore(sess, save_path=last_chk_path)
        print("Restored checkpoint from:", last_chk_path)
    except:
        print("Failed to restore checkpoint. ")
        quit()
    predict_logits=sess.run([cifar_10_model.logits], 
                            feed_dict={cifar_10_model.is_training: False, 
                                       cifar_10_model.X: cifar_10_data.test_data, 
                                       cifar_10_model.y: cifar_10_data.test_labels})[0]
    predict=np.argmax(predict_logits,axis=1)
    conf_mx=sklearn.metrics.confusion_matrix(cifar_10_data.test_labels,predict)
    print('confusion matrix:',conf_mx)
    np.fill_diagonal(conf_mx,0)
    plt.matshow(conf_mx,cmap=plt.cm.gray)
    plt.show()
