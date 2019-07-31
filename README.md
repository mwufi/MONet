# TF, MONet?!

MONet is an algorithm for understanding visual scenes by decomposing them into objects.

Right now, we handle a synthetic dataset of shapes (inspired by the COBRA paper). However, in the future, we'll be sure to train on CLEVR and other 3D datasets!

## Generation

To test out MONet on a subset of the Pygame dataset, first [install pip](idk), then download a pretrained checkpoint, or train your own. I trained:

* [all_shapes](url): Trained on the full Pygame dataset, which is generated. Colors vary from blue to red.

*more instructions about `monet_infer.py`*

## Training

MONet can train on the shapes dataset in ~1-2 days on a single V100 GPU. To train, first [follow the setup instructions for MONet], using the develop environment. Then download the [Simple Shapes Datasets](url) as TFRecords.

To test that training works, run from the root of the repo directory:

```bash
python monet_train.py --hparams='{"train_data_path":"/path/to/nsynth-train.tfrecord", "train_root_dir":"/tmp/gansynth/train"}'
```

This will run the model with suitable hyperparmeters for quickly testing training (which you can find in `model.py`). You can train with this config by adding it as a flag:

```bash
python monet_train.py --config=shape_el --hparams='{"train_data_path":"/path/to/monet-train.tfrecord", "train_root_dir":"/tmp/monet/train"}'
```