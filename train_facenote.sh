#!/usr/bin/bash

export PYTHONPATH=$PYTHONPATH:./facenote/

python3 -m facenote.facenet_train_classifier \
       	--logs_base_dir ~/logs/facenet/ \
	--models_base_dir ~/models/facenet/ \
	--data_dir ~/datasets/facescrub_mtcnnpy_182 \
	--image_size 160 \
	--model_def inception_resnet_v1 \
	--lfw_dir  ~/datasets/lfw_mtcnnalign_160 \
	--optimizer RMSPROP \
	--learning_rate -1 \
	--max_nrof_epochs 200 \
	--keep_probability 0.8 \
	--random_crop \
	--random_flip \
	--learning_rate_schedule_file ../data/learning_rate_schedule_classifier.txt \
	--weight_decay 5e-5 \
	--center_loss_factor 1e-4 \
	--center_loss_alfa 0.9 \
	--batch_size 70 \
	--epoch_size 100



