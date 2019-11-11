import pandas as pd


_lfw_root = '../data/images/'
_lfw_images = '../data/peopleDevTest.txt'
_lfw_landmarks = '../data/LFW.csv'

with open(_lfw_images) as f:
    images_lines = f.readlines()[1:]
test_image_list = pd.DataFrame(columns = ['IMAGE_NAME', 'IDENTITY', 'INDEX'])
count = 0
for line in images_lines:
    identity, num_of_images = line.replace('\n', '').split('\t')
    num_of_images = int(num_of_images)
    if num_of_images == 1: continue
    count += 1
    for i in range(1, num_of_images):
        image_name = identity + '/' + identity + '_' + '{:04}.jpg'.format(i)
        test_image_list = test_image_list.append(pd.DataFrame({'IMAGE_NAME': image_name, 'IDENTITY': identity, 'INDEX':i}, index=["0"]), ignore_index=True)
print(count)
print(test_image_list.shape)
test_image_list.to_csv('../data/evaluationImages.csv', index=False)

