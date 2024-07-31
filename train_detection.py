from ultralytics import YOLO

model_version = 'yolov9e.pt'
model_data_path = './dataset/data.yaml'
save_folder = './yolov9e_80ep_formaug'

start = False

if start:
    # Start training
    model = YOLO(model_version)

    # Training.
    results = model.train(
       data=model_data_path,
       imgsz=960,
       epochs=80,
       batch=8,
       name=save_folder,
       device='0',
       augment=True,exist_ok=True,resume=True)
else:
    # Load trainied model
    model = YOLO(save_folder + "/weights/last.pt")
    print('Model loaded...')
    # Resume Training.
    results = model.train(resume=True)