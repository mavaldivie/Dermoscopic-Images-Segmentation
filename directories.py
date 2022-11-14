import os

ISIC_2017 = 'isic_2017'
MIXED = 'old'
PH2 = 'PH2'
AUGMENTED = 'augmented_isic_2017'
base_directory = os.path.join('dataset', AUGMENTED)

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

ph2_directory = os.path.join('dataset', PH2)
ph2_images_directory = os.path.join(ph2_directory, 'images')
ph2_masks_directory = os.path.join(ph2_directory, 'masks')
ph2_images = [os.path.join(ph2_images_directory, i) for i in sorted(os.listdir(ph2_images_directory))]
ph2_masks = [os.path.join(ph2_masks_directory, i) for i in sorted(os.listdir(ph2_masks_directory))]


augmented_base_directory = os.path.join('dataset', AUGMENTED)

augmented_train_directory = os.path.join(augmented_base_directory, 'train')
augmented_validation_directory = os.path.join(augmented_base_directory, 'validation')
augmented_test_directory = os.path.join(augmented_base_directory, 'test')

augmented_train_images_directory = os.path.join(augmented_train_directory, 'images')
augmented_train_masks_directory = os.path.join(augmented_train_directory, 'masks')
augmented_validation_images_directory = os.path.join(augmented_validation_directory, 'images')
augmented_validation_masks_directory = os.path.join(augmented_validation_directory, 'masks')
augmented_test_images_directory = os.path.join(augmented_test_directory, 'images')
augmented_test_masks_directory = os.path.join(augmented_test_directory, 'masks')
