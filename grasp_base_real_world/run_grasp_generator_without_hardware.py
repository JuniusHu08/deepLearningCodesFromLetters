from grasp_generator_without_hardware import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        saved_model_path='saved_data/cornell_rgbd_iou_0.96',
        visualize=True
    )
    generator.load_model()
    generator.run()
