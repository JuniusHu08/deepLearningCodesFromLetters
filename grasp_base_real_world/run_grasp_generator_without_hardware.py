from grasp_generator_without_hardware import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        saved_model_path='/home/junhaohu/pythonCode/train_res/241220_1711_/epoch_319_iou_0.97_statedict.pt',
        # saved_model_path='/home/junhaohu/pythonCode/train_res/trained-models/jacquard-d-grconvnet3-drop0-ch32/epoch_50_iou_0.94',
        visualize=True
    )
    generator.load_model()
    generator.run()
