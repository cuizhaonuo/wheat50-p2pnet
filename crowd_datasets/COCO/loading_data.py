"""Build train and validation datasets for COCO wheat point annotations."""

from .coco_wheat_points import CocoWheatPoints


def loading_data(data_root, args):
    train_set = CocoWheatPoints(
        data_root=data_root,
        annotation_file=args.coco_train_annotations,
        image_dir=args.coco_train_images,
        category_name=args.coco_category_name,
        train=True,
        crop_size=args.coco_crop_size,
        crops_per_image=args.coco_crops_per_image,
        scale_range=(args.coco_scale_min, args.coco_scale_max),
        horizontal_flip_probability=args.coco_hflip_probability,
        copy_paste_probability=args.coco_copy_paste_probability,
        copy_paste_max_instances=args.coco_copy_paste_max_instances,
        copy_paste_placement_attempts=(
            args.coco_copy_paste_placement_attempts
        ),
    )
    val_set = CocoWheatPoints(
        data_root=data_root,
        annotation_file=args.coco_val_annotations,
        image_dir=args.coco_val_images,
        category_name=args.coco_category_name,
        train=False,
    )
    return train_set, val_set
