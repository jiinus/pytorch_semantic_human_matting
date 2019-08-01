import json
import os

img_dir = '../val2017/'
instance_dir = '../annotations/instances_val2017.json'

with open(instance_dir, 'r') as f:
    instance = json.load(f)

count = 0
for anno in instance['annotations']:
    if anno['category_id'] == 1:
        id = str(anno['image_id'])
        filename = '0' * (12 - len(id)) + id + '.jpg'
        file_path = os.path.join(img_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            count += 1

print('%d imgs removed' % (count,))
