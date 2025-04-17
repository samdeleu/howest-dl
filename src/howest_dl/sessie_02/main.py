from howest_dl.sessie_02.malaria import (
    read_malaria_input,
    preprocess_data,
    build_malaria_model,
    train_malaria_model,
)

from howest_dl.sessie_02.model_explorer import (
    model_explorer,
)

from howest_dl.sessie_02.helpers import (
    display_title,
    collect_accuracy,
    print_binary_metrics,
)

if __name__ == '__main__':
    image_size = 50
    sample_size = 1000
    X_train, y_train, X_test, y_test = read_malaria_input(sample_size=sample_size, image_size=image_size)
    print(type(X_train))

    skip = True
    if not skip:
        preprocess_data()
        malaria_model = build_malaria_model(input_shape=(image_size, image_size, 3))
        print(malaria_model.summary())

        history, timing = train_malaria_model(
            model=malaria_model,
            X_train_input=X_train, y_train_input=y_train,
            description="First attempt"
        )
        print(malaria_model.metrics_names)
        # Metrics on the training set
        print_binary_metrics(model=malaria_model, X_test_input=X_train, y_test_input=y_train, decision_boundary=0.5,
                             title="Training Set")
        # Metrics on the test set
        print_binary_metrics(model=malaria_model, X_test_input=X_test, y_test_input=y_test, decision_boundary=0.5,
                             title="Test Set")
        collect_accuracy(model=malaria_model, X_train_input=X_train, y_train_input=y_train, X_test_input=X_test,
                         y_test_input=y_test, decision_boundary=0.5)

        # Metrics on the training set
        print_binary_metrics(model=malaria_model, X_test_input=X_train, y_test_input=y_train, decision_boundary=0.95,
                             title="Training Set")
        # Metrics on the test set
        print_binary_metrics(model=malaria_model, X_test_input=X_test, y_test_input=y_test, decision_boundary=0.95,
                             title="Test Set")
        collect_accuracy(model=malaria_model, X_train_input=X_train, y_train_input=y_train, X_test_input=X_test,
                         y_test_input=y_test, decision_boundary=0.95)

        y_pred_proba = malaria_model.predict(X_test).flatten()
        y_pred_class = (y_pred_proba >= 0.95).astype(int)

        X_test_fp = X_test[(y_test == 0) & (y_pred_class == 1)]
        X_test_fn = X_test[(y_test == 1) & (y_pred_class == 0)]

        display_title("Examples of False Negatives")
        for i in range(min([3, len(X_test_fn)])):
            print(X_test_fn[i])

    image_shape = (image_size, image_size, 3)
    # Results: Name, threshold, "accuracy on training set", "accuracy on test set"
    malaria_models = [
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 01",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 128), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 128), "B", ("A", "relu"),
        #         ("D", 64), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 02",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 16), "B", ("A", "relu"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 64), "B", ("A", "relu"),
        #         ("D", 32), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 03",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 128), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("P", "avg"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("P", "avg"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("P", "avg"),
        #     ],
        #     classifier=[
        #         ("D", 128), "B", ("A", "relu"),
        #         ("D", 64), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 04",
        #     input_shape=(image_size, image_size, 3),
        #     feature_extractor=[
        #         ("C", 128), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("P", "max"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("P", "max"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 128), "B", ("A", "relu"),
        #         ("D", 64), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 05",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 128), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("P", "max"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("P", "max"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 128), "B", ("A", "relu"),
        #         ("D", 64), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 06",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 256), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("P", "max"),
        #         ("C", 128), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("P", "max"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("P", "max"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 128), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("D", 64), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 07",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 16), "B", ("A", "relu"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 64), "B", ("A", "relu"),
        #         ("D", 32), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 08",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 16), "B", ("A", "relu"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 64), "B", ("A", "relu"),
        #         ("D", 32), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 09",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 16), "B", ("A", "relu"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 64), "B", ("A", "relu"),
        #         ("D", 32), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 10",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 16), "B", ("A", "relu"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 64), "B", ("A", "relu"),
        #         ("D", 32), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 11",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 16), "B", ("A", "relu"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 64), "B", ("A", "relu"),
        #         ("D", 32), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 12",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 16), "B", ("A", "relu"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 64), "B", ("A", "relu"),
        #         ("D", 32), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 13",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 16), "B", ("A", "relu"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 64), "B", ("A", "relu"),
        #         ("D", 32), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 14",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 16), "B", ("A", "relu"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("C", 64), "B", ("A", "relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 64), "B", ("A", "relu"),
        #         ("D", 32), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 15",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 8), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 16), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 1024), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("D", 1024), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 16",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 8), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 16), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 32), "B", ("A", "relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 1024), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("D", 1024), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 17",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 8), "B", ("A", "leaky_relu"),
        #         ("P", "max"),
        #         ("C", 16), "B", ("A", "leaky_relu"),
        #         ("P", "max"),
        #         ("C", 32), "B", ("A", "leaky_relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 1024), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("D", 1024), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        # model_explorer(
        #     name=f"Exploration ({image_size}x{image_size}) 18",
        #     input_shape=image_shape,
        #     feature_extractor=[
        #         ("C", 128), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 128), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 256), "B", ("A", "relu"),
        #         ("P", "max"),
        #         ("C", 128), "B", ("A", "relu"),
        #         ("P", "max"),
        #     ],
        #     classifier=[
        #         ("D", 1024), "B", ("A", "relu"),
        #         ("Dr", 0.3),
        #         ("D", 128), "B", ("A", "relu"),
        #         ("D", 1),
        #         ("A", "sigmoid"),
        #     ]
        # ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 19",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 128), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 256), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 256), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 128), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 1024), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("D", 512), "B", ("A", "relu"),
                ("D", 64), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),

    ]

    collected_results = {}
    for i, m in enumerate(malaria_models):
        if m is None:
            continue

        print(f"Session: {i}-{getattr(m, 'howest', 'xx')}")
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        y_train_copy = y_train.copy()
        y_test_copy = y_test.copy()
        training_history = train_malaria_model(
            model=m,
            X_train_input=X_train_copy, y_train_input=y_train_copy,
            description=f"{i}-{getattr(m, 'howest', 'xx')}",
            verbose=1,
        )

        for threshold in [0.5, 0.9, 0.99]:
            if collected_results.get(threshold) is None:
                collected_results[threshold] = []

            collected_results[threshold].append(
                collect_accuracy(
                    model=m,
                    X_train_input=X_train_copy, y_train_input=y_train_copy,
                    X_test_input=X_test_copy, y_test_input=y_test_copy,
                    decision_boundary=threshold,
                    verbose=0,
                )
            )

    for threshold, collected_results in collected_results.items():
        print("Treshold:", threshold)
        for n, t, train_a, test_a in collected_results:
            print(f"{n}\tthreshold:\t{t:.2f}\tTrain:\t{train_a:.2f}\tTest:\t{test_a:.2f})")

    """
    Summary results
    Treshold: 0.5
    Exploration (50x50) 01	threshold:	0.50	Train:	98.00	Test:	93.10)
    Exploration (50x50) 02	threshold:	0.50	Train:	98.10	Test:	92.55)
    Exploration (50x50) 03	threshold:	0.50	Train:	95.75	Test:	91.85)
    Exploration (50x50) 04	threshold:	0.50	Train:	57.95	Test:	55.95)
    Exploration (50x50) 05	threshold:	0.50	Train:	50.15	Test:	50.00)
    Exploration (50x50) 06	threshold:	0.50	Train:	51.45	Test:	50.90)
    Exploration (50x50) 07	threshold:	0.50	Train:	98.50	Test:	94.30)
    Exploration (50x50) 08	threshold:	0.50	Train:	98.50	Test:	93.80)
    Exploration (50x50) 09	threshold:	0.50	Train:	98.25	Test:	93.65)
    Exploration (50x50) 10	threshold:	0.50	Train:	98.70	Test:	94.40)
    Exploration (50x50) 11	threshold:	0.50	Train:	98.55	Test:	94.00)
    Exploration (50x50) 12	threshold:	0.50	Train:	98.05	Test:	93.35)
    Exploration (50x50) 13	threshold:	0.50	Train:	98.50	Test:	93.10)
    Exploration (50x50) 14	threshold:	0.50	Train:	98.50	Test:	93.60)
    Exploration (50x50) 15	threshold:	0.50	Train:	98.20	Test:	91.10)
    Exploration (50x50) 16	threshold:	0.50	Train:	97.60	Test:	89.25)
    Exploration (50x50) 17	threshold:	0.50	Train:	98.00	Test:	91.40)
    Exploration (50x50) 18	threshold:	0.50	Train:	98.20	Test:	93.60)
    Exploration (50x50) 19	threshold:	0.50	Train:	98.15	Test:	94.45)

    Treshold: 0.9
    Exploration (50x50) 01	threshold:	0.90	Train:	96.55	Test:	90.35)
    Exploration (50x50) 02	threshold:	0.90	Train:	96.35	Test:	89.15)
    Exploration (50x50) 03	threshold:	0.90	Train:	89.55	Test:	86.65)
    Exploration (50x50) 04	threshold:	0.90	Train:	50.00	Test:	50.00)
    Exploration (50x50) 05	threshold:	0.90	Train:	50.00	Test:	50.00)
    Exploration (50x50) 06	threshold:	0.90	Train:	50.00	Test:	50.00)
    Exploration (50x50) 07	threshold:	0.90	Train:	98.15	Test:	91.85)
    Exploration (50x50) 08	threshold:	0.90	Train:	96.80	Test:	91.50)
    Exploration (50x50) 09	threshold:	0.90	Train:	95.60	Test:	90.05)
    Exploration (50x50) 10	threshold:	0.90	Train:	98.60	Test:	92.20)
    Exploration (50x50) 11	threshold:	0.90	Train:	97.80	Test:	91.40)
    Exploration (50x50) 12	threshold:	0.90	Train:	97.25	Test:	91.90)
    Exploration (50x50) 13	threshold:	0.90	Train:	96.65	Test:	89.30)
    Exploration (50x50) 14	threshold:	0.90	Train:	96.05	Test:	89.60)
    Exploration (50x50) 15	threshold:	0.90	Train:	97.90	Test:	89.15)
    Exploration (50x50) 16	threshold:	0.90	Train:	97.55	Test:	88.50)
    Exploration (50x50) 17	threshold:	0.90	Train:	97.95	Test:	91.25)
    Exploration (50x50) 18	threshold:	0.90	Train:	98.00	Test:	94.10)
    Exploration (50x50) 19	threshold:	0.90	Train:	97.60	Test:	93.95)

    Treshold: 0.99
    Exploration (50x50) 01	threshold:	0.99	Train:	86.05	Test:	80.35)
    Exploration (50x50) 02	threshold:	0.99	Train:	73.40	Test:	70.00)
    Exploration (50x50) 03	threshold:	0.99	Train:	75.10	Test:	74.90)
    Exploration (50x50) 04	threshold:	0.99	Train:	50.00	Test:	50.00)
    Exploration (50x50) 05	threshold:	0.99	Train:	50.00	Test:	50.00)
    Exploration (50x50) 06	threshold:	0.99	Train:	50.00	Test:	50.00)
    Exploration (50x50) 07	threshold:	0.99	Train:	89.40	Test:	82.80)
    Exploration (50x50) 08	threshold:	0.99	Train:	84.60	Test:	79.05)
    Exploration (50x50) 09	threshold:	0.99	Train:	72.45	Test:	69.60)
    Exploration (50x50) 10	threshold:	0.99	Train:	94.00	Test:	86.90)
    Exploration (50x50) 11	threshold:	0.99	Train:	90.95	Test:	83.95)
    Exploration (50x50) 12	threshold:	0.99	Train:	89.85	Test:	84.90)
    Exploration (50x50) 13	threshold:	0.99	Train:	79.15	Test:	75.50)
    Exploration (50x50) 14	threshold:	0.99	Train:	75.65	Test:	73.20)
    Exploration (50x50) 15	threshold:	0.99	Train:	94.05	Test:	85.85)
    Exploration (50x50) 16	threshold:	0.99	Train:	97.05	Test:	86.05)
    Exploration (50x50) 17	threshold:	0.99	Train:	96.25	Test:	89.00)
    Exploration (50x50) 18	threshold:	0.99	Train:	96.25	Test:	92.55)
    Exploration (50x50) 19	threshold:	0.99	Train:	93.50	Test:	89.35) 
    """