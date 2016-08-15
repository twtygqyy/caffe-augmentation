# Caffe Augmentation Extension
This is a modified caffe fork (version of 2016/8/15) with ImageData layer data augmentation, which is based on:

[@kevinlin311tw](https://github.com/kevinlin311tw)'s [caffe-augmentation](https://github.com/kevinlin311tw/caffe-augmentation),
[@ChenlongChen](https://github.com/ChenglongChen)'s [caffe-windows](https://github.com/ChenglongChen/caffe-windows), 
[@ShaharKatz](https://github.com/ShaharKatz)'s [Caffe-Data-Augmentation](https://github.com/ShaharKatz/Caffe-Data-Augmentation), 
[@senecaur](https://github.com/senecaur)'s [caffe-rta](https://github.com/senecaur/caffe-rta).
[@kostyaev](https://github.com/kostyaev)'s [caffe-augmentation](https://github.com/kostyaev/caffe-augmentation)


min_side_min nad min_side_max are added for random cropping while keeping the aspect ratio, as mentioned in "Deep Residual Learning for Image Recognition"(http://arxiv.org/abs/1512.03385)

and all functions
* min_side - resize and crop preserving aspect ratio, default 0 (disabled);
* max_rotation_angle - max angle for an image rotation, default 0;
* contrast_brightness_adjustment - enable/disable contrast adjustment, default false;
* smooth_filtering - enable/disable smooth filterion, default false;
* min_contrast - min contrast multiplier (min [alpha](http://docs.opencv.org/2.4/doc/tutorials/core/basic_linear_transform/basic_linear_transform.html)), default 0.8;
* max_contrast - min contrast multiplier (max [alpha](http://docs.opencv.org/2.4/doc/tutorials/core/basic_linear_transform/basic_linear_transform.html)), default 1.2;
* max_brightness_shift - max brightness shift in positive and negative directions ([beta](http://docs.opencv.org/2.4/doc/tutorials/core/basic_linear_transform/basic_linear_transform.html)), default 5;
* max_smooth - max smooth multiplier, default 6;
* max_color_shift - max color shift along RGB axes
* apply_probability - how often every transformation should be applied, default 0.5;
* debug_params - enable/disable printing tranformation parameters, default false;
 from [@kostyaev](https://github.com/kostyaev)'s [caffe-augmentation](https://github.com/kostyaev/caffe-augmentation) are kept with slightly modifications:

## How to use
You could specify your network prototxt as:

    layer {
    name: "data"
    type: "ImageData"
    top: "data"
    top: "label"
    include {
      phase: TRAIN
    }
    transform_param {
        mirror: true
        contrast_brightness_adjustment: true
        smooth_filtering: true
        min_side_min: 256
        min_side_max: 480
        crop_size: 224
        mean_file: "imagenet_mean.binaryproto"
        min_contrast: 0.8
        max_contrast: 1.2
        max_smooth: 6
        apply_probability: 0.5
        max_color_shift: 20
        debug_params: false
    }
    image_data_param {
      source: "train_list.txt"
      batch_size: 64
      shuffle: true
    }
    }

while in testing phase:

    layer {
    name: "data"
    type: "ImageData"
    top: "data"
    top: "label"
    include {
      phase: TRAIN
    }
    transform_param {
        mirror: false
        min_side: 256
        crop_size: 224
        mean_file: "imagenet_mean.binaryproto"
    }
    image_data_param {
      source: "test_list.txt"
      batch_size: 32
    }
    }
