def predict_image():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    url = "/Users/joshbenzon/Documents/CompSci/cs1430_python/hw5/code/test.JPG"
    path = tf.keras.utils.get_file(fname="~water~", origin=url)

    img = tf.keras.utils.load_img(path)

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print("This image msot loikely belongs to {} with a {:.2f} percent confidence."
          .format(class_names[np.argmax(score)], 100 * np.max(score)))
