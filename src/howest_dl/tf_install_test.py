def tf_install_test_2():
    print("222 Checking if tensorflow is installed")
    import tensorflow as tf

    print("-------------- CPU setup --------------")
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))

    print("-------------- GPU setup --------------")
    print(tf.config.list_physical_devices('GPU'))


if __name__ == "__main__":
    tf_install_test_2()
