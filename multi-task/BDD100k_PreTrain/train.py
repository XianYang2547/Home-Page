import sys
sys.path.insert(0, "/home/xianyang/Desktop/YOLOv8-multi-task")

from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8-multi-m.yaml', task='multi')  # build a new model from YAML
model = YOLO('/home/xianyang/Desktop/YOLOv8-multi-task/BDD100k_PreTrain/runs/multi/TRAIN_Pro36+32/weights/best.pt', task='multi')  # build a new model from YAML


# Train the model
model.train(data='bdd100k.yaml', batch=24, epochs=32, imgsz=640,
            device=[0], name='TRAIN_Pro32', val=True, task='multi', classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            combine_class=[], single_cls=False)

# 36轮
'''
36 epochs completed in 16.088 hours.
Optimizer stripped from runs/multi/TRAIN_Pro/weights/last.pt, 58.8MB
Optimizer stripped from runs/multi/TRAIN_Pro/weights/best.pt, 58.8MB
Validating runs/multi/TRAIN_Pro/weights/best.pt...
Ultralytics YOLOv8.0.105 🚀 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (NVIDIA GeForce RTX 4090, 24209MiB)
YOLOv8-multi-m summary (fused): 305 layers, 29182252 parameters, 0 gradients
100%|██████████| 209/209 [01:20<00:00,  2.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50   mAP50-95
                   all      10000     185578      0.047      0.739      0.532      0.301
                   dog      10000      13265     0.0547      0.806      0.625      0.313
                 rider      10000        649     0.0222      0.713      0.459      0.229
                   car      10000     102540      0.128      0.868      0.792      0.492
                   bus      10000       1597     0.0328      0.907      0.634       0.48
                 truck      10000       4247     0.0355      0.918      0.649      0.468
                  bike      10000       1007     0.0199      0.755      0.451      0.227
                 motor      10000        452     0.0187       0.75      0.453      0.223
         traffic light      10000      26891     0.0897      0.769      0.599      0.232
          traffic sign      10000      34915     0.0674      0.841      0.655      0.349
                 train      10000         15    0.00139     0.0667     0.0053    0.00053
             Lane line     pixacc      0.992     subacc      0.843        IoU      0.282       mIoU      0.637
Speed: 0.1ms preprocess, 1.5ms inference, 0.0ms loss, 0.2ms postprocess per image
'''
# 再训练32轮
'''
32 epochs completed in 9.377 hours.
Optimizer stripped from runs/multi/TRAIN_Pro32/weights/last.pt, 58.8MB
Optimizer stripped from runs/multi/TRAIN_Pro32/weights/best.pt, 58.8MB
Validating runs/multi/TRAIN_Pro32/weights/best.pt...
Ultralytics YOLOv8.0.105 🚀 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (NVIDIA GeForce RTX 4090, 24209MiB)
YOLOv8-multi-m summary (fused): 305 layers, 29182252 parameters, 0 gradients
100%|██████████| 209/209 [01:19<00:00,  2.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50   mAP50-95
                   all      10000     185578       0.05      0.747      0.548      0.312
                   dog      10000      13265      0.058      0.815      0.644      0.329
                 rider      10000        649      0.024      0.727      0.479      0.245
                   car      10000     102540      0.137      0.873        0.8      0.498
                   bus      10000       1597     0.0339      0.909      0.646      0.491
                 truck      10000       4247     0.0384      0.926      0.663      0.481
                  bike      10000       1007     0.0217      0.751      0.481      0.238
                 motor      10000        452     0.0217      0.772      0.482       0.23
         traffic light      10000      26891      0.094      0.781      0.614      0.241
          traffic sign      10000      34915     0.0699      0.853      0.672      0.361
                 train      10000         15   0.000894     0.0667    0.00108   0.000216
             Lane line     pixacc      0.992     subacc      0.859        IoU      0.286       mIoU      0.639
Speed: 0.1ms preprocess, 1.5ms inference, 0.0ms loss, 0.2ms postprocess per image
'''
# 再训练32轮 共100轮
'''32 epochs completed in 9.343 hours.
Optimizer stripped from runs/multi/TRAIN_Pro32/weights/last.pt, 58.8MB
Optimizer stripped from runs/multi/TRAIN_Pro32/weights/best.pt, 58.8MB

Validating runs/multi/TRAIN_Pro32/weights/best.pt...
Ultralytics YOLOv8.0.105 🚀 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (NVIDIA GeForce RTX 4090, 24209MiB)
YOLOv8-multi-m summary (fused): 305 layers, 29182252 parameters, 0 gradients
100%|██████████| 209/209 [01:20<00:00,  2.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50   mAP50-95
                   all      10000     185578     0.0515      0.755      0.553      0.317
                   dog      10000      13265     0.0594       0.82      0.651      0.334
                 rider      10000        649     0.0239      0.718      0.489      0.256
                   car      10000     102540      0.142      0.874      0.803      0.501
                   bus      10000       1597     0.0353      0.908      0.653      0.497
                 truck      10000       4247     0.0397      0.923      0.666      0.484
                  bike      10000       1007     0.0226      0.763      0.488      0.249
                 motor      10000        452     0.0224      0.774      0.482      0.234
         traffic light      10000      26891     0.0955      0.786      0.619      0.244
          traffic sign      10000      34915      0.073      0.854      0.677      0.366
                 train      10000         15    0.00176      0.133    0.00111   0.000381
             Lane line     pixacc      0.992     subacc      0.861        IoU      0.288       mIoU       0.64
Speed: 0.1ms preprocess, 1.5ms inference, 0.0ms loss, 0.2ms postprocess per image
'''