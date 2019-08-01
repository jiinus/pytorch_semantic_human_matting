import os

fg_dir = '/home/soyan/labs/seg/data/alpha/'
bg_dir = '/home/soyan/labs/seg/val2017/'

train_count = 3000
val_count = 300

with open('./data/bg_list.txt', 'w') as bg_list:
    bgs = os.listdir(bg_dir)
    bg_list.writelines([(os.path.join(bg_dir, bg) + '\n') for bg in bgs])
with open('./data/train_fg_list.txt', 'w') as train_fg_list:
    with open('./data/val_fg_list.txt', 'w') as val_fg_list:
        fgs = sorted(os.listdir(fg_dir))
        train_fg_list.writelines([(os.path.join(fg_dir, fgs[i]) + '\n') for i in range(0, train_count)])
        val_fg_list.writelines([(os.path.join(fg_dir, fgs[i]) + '\n') for i in range(train_count, train_count + val_count)])