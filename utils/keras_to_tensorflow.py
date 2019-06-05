import tensorflow as tf
from keras import backend as K
from pathlib import Path
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

def keras_to_tensorflow_model(model, output_model_path, output_nodes_prefix, save_graph_def = True, quantize = False):
    K.set_learning_phase(0)

    if str(Path(output_model_path).parent) == '.':
        output_model_path = str((Path.cwd() / output_model_path))

    output_fld = Path(output_model_path).parent
    output_model_name = Path(output_model_path).name
    output_model_stem = Path(output_model_path).stem
    output_model_pbtxt_name = output_model_stem + '.pbtxt'
    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    
    
    # TODO(amirabdi): Support networks with multiple inputs
    orig_output_node_names = [node.op.name for node in model.outputs]
    if output_nodes_prefix:
        num_output = len(orig_output_node_names)
        pred = [None] * num_output
        converted_output_node_names = [None] * num_output

        # Create dummy tf nodes to rename output
        for i in range(num_output):
            converted_output_node_names[i] = '{}{}'.format(
                output_nodes_prefix, i)
            pred[i] = tf.identity(model.outputs[i],
                                  name=converted_output_node_names[i])
    else:
        converted_output_node_names = orig_output_node_names

    sess = K.get_session()

    if save_graph_def:
        tf.train.write_graph(sess.graph.as_graph_def(), str(output_fld),
                             output_model_pbtxt_name, as_text=True)

    if quantize:
        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [],
                                               converted_output_node_names,
                                               transforms)
        constant_graph = graph_util.convert_variables_to_constants(
            sess,
            transformed_graph_def,
            converted_output_node_names)
    else:
        constant_graph = graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            converted_output_node_names)

    graph_io.write_graph(constant_graph, str(output_fld), output_model_name,
                         as_text=False)