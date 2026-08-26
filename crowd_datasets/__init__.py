# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == 'SHHA':
        from crowd_datasets.SHHA.loading_data import loading_data
        return loading_data(args.data_root)

    if args.dataset_file in {'COCO', 'COCO_WHEAT_POINTS'}:
        from crowd_datasets.COCO.loading_data import loading_data
        return loading_data(args.data_root, args)

    raise ValueError(f'Unsupported dataset_file: {args.dataset_file}')
