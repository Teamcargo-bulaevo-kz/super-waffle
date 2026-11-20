import glob, random, numpy as np, tensorflow as tf, sys, os
from tensorflow import keras
from tensorflow.keras import layers as L
random.seed(0); np.random.seed(0); tf.random.set_seed(0)
files=sorted(glob.glob(os.environ.get('NEUROCAM_DATA','data/flower_photos')+'/*/*.jpg')); random.shuffle(files)
val_files, tr_files = files[:120], files[120:]
P=96
def load(f):
    img=tf.io.decode_jpeg(tf.io.read_file(f),channels=3)
    img=tf.image.convert_image_dtype(img,tf.float32)
    s=tf.random.uniform([],0.55,1.0)
    hw=tf.cast(tf.cast(tf.shape(img)[:2],tf.float32)*s,tf.int32)
    hw=tf.maximum(hw,P)
    img=tf.image.resize(img,hw,method='area',antialias=True)
    img=tf.image.random_crop(img,[P,P,3])
    img=tf.image.random_flip_left_right(img)
    if tf.random.uniform([])<0.5: img=tf.image.rot90(img)
    return img
def gauss_kernel(sigma):
    x=tf.range(-3,4,dtype=tf.float32); k=tf.exp(-x**2/(2*sigma**2)); k=k/tf.reduce_sum(k)
    k2=k[:,None]*k[None,:]; return tf.tile(k2[:,:,None,None],[1,1,3,1])
def degrade(clean):
    x=clean
    sig=tf.random.uniform([],0.3,1.4)
    if tf.random.uniform([])<0.7:
        x=tf.nn.depthwise_conv2d(x[None],gauss_kernel(sig),[1,1,1,1],'SAME')[0]
    # Poisson-Gaussian noise in approx linear domain
    lin=tf.pow(tf.clip_by_value(x,0,1),2.2)
    shot=tf.random.uniform([],0.0,0.02); read=tf.random.uniform([],0.0005,0.012)
    std=tf.sqrt(shot*lin+read**2)
    lum=tf.random.normal(tf.shape(lin)[:2])[...,None]
    chroma=tf.random.normal(tf.shape(lin))
    mix=tf.random.uniform([],0.3,0.9)
    n=std*(mix*lum+tf.sqrt(1-mix**2)*chroma)
    # ISP denoise often leaves low-freq blotchy noise: add some blurred noise
    lin=tf.clip_by_value(lin+n,0,1)
    x=tf.pow(lin+1e-8,1/2.2)
    if tf.random.uniform([])<0.6:
        q=tf.random.uniform([],65,96,dtype=tf.int32)
        x=tf.image.adjust_jpeg_quality(tf.clip_by_value(x,0,1),q)
    return tf.clip_by_value(x,0,1), clean
def ds(fs,train=True):
    d=tf.data.Dataset.from_tensor_slices(fs)
    if train: d=d.repeat().shuffle(4000)
    return d.map(load,num_parallel_calls=8).map(degrade,num_parallel_calls=8).batch(16).prefetch(4)
def model(shape=(None,None,3)):
    i=L.Input(shape,batch_size=1 if shape[0] else None,name='input')
    e1=L.Conv2D(32,3,padding='same',activation='relu')(i)
    e1=L.Conv2D(32,3,padding='same',activation='relu')(e1)
    h=L.Conv2D(48,3,strides=2,padding='same',activation='relu')(e1)
    for _ in range(3):
        r=L.Conv2D(48,3,padding='same',activation='relu')(h)
        r=L.Conv2D(48,3,padding='same')(r)
        h=L.Add()([h,r])
    u=L.Conv2D(128,1,padding='same')(h)
    u=L.Lambda(lambda t: tf.nn.depth_to_space(t,2))(u)
    f=L.Add()([u,e1])
    f=L.Conv2D(32,3,padding='same',activation='relu')(f)
    o=L.Conv2D(3,3,padding='same')(f)
    o=L.Add(name='output')([i,o])
    return keras.Model(i,o)
if __name__=='__main__':
    steps=int(sys.argv[1])
    m=model(); print(m.count_params())
    sched=keras.optimizers.schedules.CosineDecay(2e-3,steps,alpha=0.02)
    m.compile(keras.optimizers.Adam(sched),loss='mae')
    vd=ds(val_files,False).take(30).cache()
    m.fit(ds(tr_files),steps_per_epoch=steps//10,epochs=10,validation_data=vd,verbose=2)
    m.save_weights(os.environ.get('NEUROCAM_WEIGHTS','neuro.weights.h5'))
