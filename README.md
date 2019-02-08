## Welcome to FaceNote Pages

This Project focus on computer vision and artificial intelligence for face searching.

- OpenCV
- MTCNN
- FaceNet
- Python3
- TensorFlow 

The project has below components

1. Face detect and search validation

## FaceNote
### Install Requirments
```
pip3 install -r requirments.txt
```

### Download Training Data
```
python3 facescrub_download.py
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
```

### Align Images
```
python3 align_dataset_mtcnn.py ~/datasets/lfw/ ~/datasets/lfw_mtcnnalign_160 --image_size 160 --margin 32
python3 align_dataset_mtcnn.py ~/datasets/facescrub/ ~/datasets/facescrub_mtcnnpy_182 --image_size 182 --margin 44
```

### Training classifier
```
python3 facenet_train_classifier.py --logs_base_dir ~/logs/facenet/ --models_base_dir ~/models/facenet/ --data_dir ~/datasets/facescrub_mtcnnpy_182 --image_size 160 --model_def inception_resnet_v1 --lfw_dir  ~/datasets/lfw_mtcnnalign_160 --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 200 --keep_probability 0.8 --random_crop --random_flip --learning_rate_schedule_file ../data/learning_rate_schedule_classifier.txt --weight_decay 5e-5 --center_loss_factor 1e-4 --center_loss_alfa 0.9 --batch_size 70 --epoch_size 100
```
If you have a pre-trained model, please add the --pretrained_model in the above command.

--pretrained_model $pretrained_model_name

For example:

~/tmp/models/facenet/20170223-125024/model-20170223-125024.ckpt-0


### Image Embedding


### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/overwindows/renaissance/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

### Raspberry
```
sudo mount -t cifs -o username=Minerva,password=sesame //192.168.8.1/sda2 /mnt/net_disk
sudo mount -o uid=pi,gid=pi /dev/sda1 /mnt/usb_flash
[OpenCV](https://www.cnblogs.com/zjutlitao/p/8261688.html)
```
