from grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=830112070066,
        saved_model_path='/home/junhaohu/pythonCode/train_res/241220_1711_/epoch_62_iou_0.97_statedict.pt',
        visualize=True
    )
    generator.load_model()
    generator.run()
