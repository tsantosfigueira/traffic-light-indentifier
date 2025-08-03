from ultralytics import YOLO

def run():

    model = YOLO("yolov8n.pt")

    model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0
    )

    result = model.predict(source='valid/images/30746_png.rf.46d7c85de51ba1630d7151ca120e5906.jpg', save=True)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # necessário apenas para Windows
    run()