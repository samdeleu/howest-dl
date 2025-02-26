def tf_install_test():
    print("Checking if tensorflow is installed")
    import tensorflow as tf

    print("-------------- CPU setup --------------")
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))

    print("-------------- GPU setup --------------")
    print(tf.config.list_physical_devices('GPU'))

def main() -> None:
    print("Hello from howest-dl -learning!")
    print("-------------------------------")

    tf_install_test()

