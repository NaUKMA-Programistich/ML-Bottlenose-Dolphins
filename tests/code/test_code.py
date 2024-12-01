from PIL import Image

from fins_training import val_transform


def test_yolo_model_loading(loaded_yolo_model):
    """
    Test that the YOLO model is loaded successfully.

    Args:
        loaded_yolo_model (YOLO): The loaded YOLO model fixture.
    """
    assert loaded_yolo_model is not None, "YOLO model failed to load."


def test_val_transform():
    """
    Test that the validation transform correctly processes an image.

    Creates a dummy RGB image, applies the validation transform,
    and asserts that the transformed image has the expected shape.
    """
    img = Image.new("RGB", (224, 224))
    transformed_img = val_transform(img)
    assert transformed_img.shape == (3, 224, 224), (
        f"Transformed image shape is {transformed_img.shape}, expected (3, 224, 224)."
    )
