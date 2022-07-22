import tensorflow as tf
import pdb
demo_data_p = "data/v1.0_waymo_block_nerf_mission_bay_train.tfrecord-00000-of-01063"
filenames = [demo_data_p]
dataset = tf.data.TFRecordDataset(filenames[0])

for raw_record in dataset.take(10):
  print(repr(raw_record))

pdb.set_trace()

# Read the data back out.
def decode_fn(record_bytes):
  return tf.io.parse_single_example(
      # Data
      record_bytes,

      # Schema
      {"image_hash": tf.io.FixedLenFeature([], dtype=tf.float32),
       "cam_idx": tf.io.FixedLenFeature([], dtype=tf.float32)}
  )

