import tensorflow.compat.v1 as tf1
import tensorflow as tf
import os

tf1.disable_eager_execution()

def load_and_convert_to_h5(saved_model_dir, output_h5_path):
    # Load the SavedModel
    with tf1.Session() as sess:
        tf1.saved_model.loader.load(sess, [tf1.saved_model.tag_constants.SERVING], saved_model_dir)
        
        # Get the graph
        graph = tf1.get_default_graph()
        
        # Get input tensors
        observations = graph.get_tensor_by_name("default_policy/observation:0")
        seq_lens = graph.get_tensor_by_name("default_policy/seq_lens:0")
        is_training = graph.get_tensor_by_name("default_policy/is_training:0")
        
        # Get output tensors
        action_prob = graph.get_tensor_by_name("default_policy/Exp_3:0")
        action_dist_inputs = graph.get_tensor_by_name("default_policy/sequential_6/action_out/BiasAdd:0")
        action_logp = graph.get_tensor_by_name("default_policy/cond_1/Merge:0")
        actions_0 = graph.get_tensor_by_name("default_policy/cond/Merge:0")
        
        # Create a new Keras model
        inputs = [
            tf.keras.layers.Input(tensor=observations, name="observations"),
            tf.keras.layers.Input(tensor=seq_lens, name="seq_lens"),
            tf.keras.layers.Input(tensor=is_training, name="is_training")
        ]
        outputs = [
            tf.keras.layers.Lambda(lambda x: action_prob, name="action_prob")(inputs[0]),
            tf.keras.layers.Lambda(lambda x: action_dist_inputs, name="action_dist_inputs")(inputs[0]),
            tf.keras.layers.Lambda(lambda x: action_logp, name="action_logp")(inputs[0]),
            tf.keras.layers.Lambda(lambda x: actions_0, name="actions_0")(inputs[0])
        ]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Save as .h5
        model.save(output_h5_path, include_optimizer=False)

# Use the function
load_and_convert_to_h5('checkpoint/saved1', 'model.h5')