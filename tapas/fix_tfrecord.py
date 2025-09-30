import tensorflow as tf
import os
import sys

def fix_tfrecord(input_file, output_file):
    """从TFRecord中删除prev_label_ids字段"""
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    def parse_example(example_proto):
        # 定义所有可能的特征，但不包括prev_label_ids
        feature_description = {
            'input_ids': tf.io.FixedLenFeature([], tf.string),
            'input_mask': tf.io.FixedLenFeature([], tf.string),
            'segment_ids': tf.io.FixedLenFeature([], tf.string),
            'column_ids': tf.io.FixedLenFeature([], tf.string),
            'row_ids': tf.io.FixedLenFeature([], tf.string),
            'column_ranks': tf.io.FixedLenFeature([], tf.string),
            'inv_column_ranks': tf.io.FixedLenFeature([], tf.string),
            'numeric_relations': tf.io.FixedLenFeature([], tf.string),
            'label_ids': tf.io.FixedLenFeature([], tf.string),
            'classification_class_index': tf.io.FixedLenFeature([], tf.string),
        }
        
        # 解析example
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return parsed
    
    def create_example(features):
        example = tf.train.Example()
        for key, value in features.items():
            example.features.feature[key].bytes_list.value[:] = [value.numpy()]
        return example
    
    # 读取原始TFRecord
    dataset = tf.data.TFRecordDataset(input_file, compression_type='GZIP')
    dataset = dataset.map(parse_example)
    
    # 写入新的TFRecord
    with tf.io.TFRecordWriter(output_file, options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
        for features in dataset:
            example = create_example(features)
            writer.write(example.SerializeToString())

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    fix_tfrecord(input_file, output_file)
    print(f"Processed {input_file} -> {output_file}")
