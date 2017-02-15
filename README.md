## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/overwindows/renaissance/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Setup
https://scipy.org/install.html
pip install Pillow

### Training Data
python facescrub_download.py

### Align Images
python align_dataset_mtcnn.py ~/datasets/lfw/ ~/datasets/lfw_mtcnnalign_160 --image_size 160 --margin 32

### Training
python facenet_train_classifier.py --logs_base_dir ~/logs/facenet/ --models_base_dir ~/models/facenet/ --data_dir ~/datasets/facescrub/facescrub_mtcnnpy_182 --image_size 160 --model_def models.inception_resnet_v1 --lfw_dir  ~/datasets/lfw/lfw_mtcnnalign_160 --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 80 --keep_probability 0.8 --random_crop --random_flip --learning_rate_schedule_file ../data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-4 --center_loss_alfa 0.9

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

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/overwindows/renaissance/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
