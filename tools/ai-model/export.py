import os, sys, tensorflow as tf, numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from train import model
T=int(sys.argv[1]) if len(sys.argv)>1 else 256
m=model((T,T,3)); m.load_weights(os.environ.get('NEUROCAM_WEIGHTS','neuro.weights.h5'))
m.export('neurocam_savedmodel', format='tf_saved_model')
c=tf.lite.TFLiteConverter.from_saved_model('neurocam_savedmodel')
c.optimizations=[tf.lite.Optimize.DEFAULT]; c.target_spec.supported_types=[tf.float16]
b=c.convert(); open('neurocam_enhance.tflite','wb').write(b); print('size',len(b))
it=tf.lite.Interpreter(model_content=b); it.allocate_tensors()
print(it.get_input_details()[0]['shape'], it.get_output_details()[0]['shape'])
x=np.random.rand(1,T,T,3).astype(np.float32); it.set_tensor(it.get_input_details()[0]['index'],x); it.invoke()
y=it.get_tensor(it.get_output_details()[0]['index']); print('maxdiff vs keras', np.abs(y-m(x).numpy()).max())
print(set(op['op_name'] for op in it._get_ops_details()))
