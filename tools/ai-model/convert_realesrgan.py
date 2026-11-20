# Rebuilds Real-ESRGAN realesr-general-x4v3 (SRVGGNetCompact) from a tf.js graph model and exports TFLite
import json, sys, numpy as np, tensorflow as tf
from tensorflow import keras
def load(model_dir):
    m=json.load(open(f'{model_dir}/model.json'))
    buf=open(f'{model_dir}/group1-shard1of1.bin','rb').read(); off=0; W={}
    for w in m['weightsManifest'][0]['weights']:
        cnt=int(np.prod(w['shape'])) if w['shape'] else 1; q=w.get('quantization')
        if q and q['dtype']=='float16': a=np.frombuffer(buf,np.float16,cnt,off).astype(np.float32); off+=cnt*2
        elif w['dtype']=='int32': a=np.frombuffer(buf,np.int32,cnt,off); off+=cnt*4
        else: a=np.frombuffer(buf,np.float32,cnt,off); off+=cnt*4
        W[w['name']]=a.reshape(w['shape'])
    assert off==len(buf), (off,len(buf))
    convs=[n for n in m['modelTopology']['node'] if n['op']=='_FusedConv2D']
    return W, convs
def build(model_dir, tile):
    W, convs = load(model_dir)
    inp=keras.Input((tile,tile,3),batch_size=1,name='input'); x=inp; layers=[]
    for i,n in enumerate(convs):
        k=W[n['input'][1]]; b=W[n['input'][2]]
        conv=keras.layers.Conv2D(k.shape[3],k.shape[:2],padding='same',name=f'conv{i}'); x=conv(x); layers.append((conv,[k,b]))
        if len(n['input'])>3:
            a=W[n['input'][3]]; pr=keras.layers.PReLU(shared_axes=[1,2],name=f'prelu{i}'); x=pr(x); layers.append((pr,[a.reshape(1,1,-1)]))
    # nearest x4 upsample of the input, expressed as channel concat + depth_to_space (GPU friendly)
    rep=keras.layers.Concatenate(axis=-1,name='rep')([inp]*16)
    x=keras.layers.Add(name='residual')([x,rep])
    out=keras.layers.Lambda(lambda t: tf.nn.depth_to_space(t,4),name='output')(x)
    model=keras.Model(inp,out)
    for l,w in layers: l.set_weights(w)
    return model
if __name__=='__main__':
    md,tile,out=sys.argv[1],int(sys.argv[2]),sys.argv[3]
    m=build(md,tile); print('params',m.count_params())
    import shutil; shutil.rmtree('realesr_savedmodel',ignore_errors=True)
    m.export('realesr_savedmodel',format='tf_saved_model')
    c=tf.lite.TFLiteConverter.from_saved_model('realesr_savedmodel')
    c.optimizations=[tf.lite.Optimize.DEFAULT]; c.target_spec.supported_types=[tf.float16]
    b=c.convert(); open(out,'wb').write(b); print('size',len(b))
    it=tf.lite.Interpreter(model_content=b); it.allocate_tensors()
    print(it.get_input_details()[0]['shape'],it.get_output_details()[0]['shape'],set(o['op_name'] for o in it._get_ops_details()))
