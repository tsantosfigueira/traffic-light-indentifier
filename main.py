from ultralytics import YOLO

def run():

    model = YOLO("yolov8m.pt")

    model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # necessário apenas para Windows
    run()