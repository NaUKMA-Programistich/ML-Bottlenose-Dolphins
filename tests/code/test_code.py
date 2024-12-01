from fins_training import val_transform


def test_yolo_model_loading(loaded_yolo_model):
    assert loaded_yolo_model is not None


def test_val_transform():
    from PIL import Image
    img = Image.new("RGB", (224, 224))
    transformed_img = val_transform(img)
    assert transformed_img.shape == (3, 224, 224)
