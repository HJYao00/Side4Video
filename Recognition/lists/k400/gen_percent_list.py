


def generate_data_percent_list(train_file, percent, output):
    # generate class dict
    cls_dict = {}
    with open(train_file) as f:
        num_videos = 0
        for x in f.readlines():
            cls_name = int(x.strip().split(' ')[-1])
            if cls_name not in cls_dict:
                cls_dict[cls_name] = [x]
            else:
                cls_dict[cls_name].append(x)
            num_videos += 1

    # calculate the number of each class, then sample
    sample_num = int(num_videos * percent)
    per_cls = sample_num // 400

    sample_list = []
    for name, v in cls_dict.items():
        sample_list.extend(v[:per_cls])

    # save output label list
    with open(out, 'w') as f:
        f.writelines(sample_list)


def generate_cls_percent_list(train_file, percent, output):
    # generate class dict
    cls_dict = {}
    with open(train_file) as f:
        num_videos = 0
        for x in f.readlines():
            cls_name = int(x.strip().split(' ')[-1])
            if cls_name not in cls_dict:
                cls_dict[cls_name] = [x]
            else:
                cls_dict[cls_name].append(x)
            num_videos += 1

    # calculate the number of each class
    cls_num = int(400 * percent)

    # sort the class by number of videos
    with open('train_dataper/k400_cls_num_sort.csv') as f:
        sort_classid = [int(x.strip().split(',')[0]) for x in f.readlines()]
    
    sort_classid = list(range(400))
    # sample class
    sample_list = []
    sample_cls = sort_classid[:cls_num]
    for cls in sample_cls:
        sample_list.extend(cls_dict[cls])

    # save output label list

    with open(out, 'w') as f:
        f.writelines(sample_list)

if __name__ == '__main__':
    train_file = 'kinetics_rgb_train_se320.txt'
    percent = 0.2
    out = 'train_dataper/kinetics_rgb_train_se320_{}_class_sort.txt'.format(percent)

    # generate_data_percent_list(train_file, percent, out)
    generate_cls_percent_list(train_file, percent, out)