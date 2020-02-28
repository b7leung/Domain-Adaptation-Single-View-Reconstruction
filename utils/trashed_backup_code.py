test.py


                # Append generated volumes to TensorBoard
                # Volume Visualization
                #gv = generated_volume.cpu().numpy()
                #rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir, 'test'),
                #                                                              epoch_idx)
                #test_writer.add_image('Test Sample#%02d/Volume Reconstructed' % sample_idx, rendering_views, epoch_idx)

                #gtv = ground_truth_volume.cpu().numpy()
                #rendering_views = utils.binvox_visualization.get_volume_views(gtv, os.path.join(img_dir, 'test'),
                #                                                              epoch_idx)
                #test_writer.add_image('Test Sample#%02d/Volume GroundTruth' % sample_idx, rendering_views, epoch_idx)