import os
base_directory = 'dataset'

train_directory = os.path.join(base_directory, 'train')
validation_directory = os.path.join(base_directory, 'validation')
test_directory = os.path.join(base_directory, 'test')

train_images_directory = os.path.join(train_directory, 'images')
train_masks_directory = os.path.join(train_directory, 'masks')
validation_images_directory = os.path.join(validation_directory, 'images')
validation_masks_directory = os.path.join(validation_directory, 'masks')
test_images_directory = os.path.join(test_directory, 'images')
test_masks_directory = os.path.join(test_directory, 'masks')

train_images = [os.path.join(train_images_directory, i) for i in sorted(os.listdir(train_images_directory))]
train_masks = [os.path.join(train_masks_directory, i) for i in sorted(os.listdir(train_masks_directory))]
validation_images = [os.path.join(validation_images_directory, i) for i in sorted(os.listdir(validation_images_directory))]
validation_masks = [os.path.join(validation_masks_directory, i) for i in sorted(os.listdir(validation_masks_directory))]
test_images = [os.path.join(test_images_directory, i) for i in sorted(os.listdir(test_images_directory))]
test_masks = [os.path.join(test_masks_directory, i) for i in sorted(os.listdir(test_masks_directory))]