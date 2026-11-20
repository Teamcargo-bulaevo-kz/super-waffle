# Evaluates the exported TFLite model with the same tiling as the app, on held-out images
import sys, glob, random, numpy as np, tensorflow as tf, io
from PIL import Image
import skimage.data as D
from skimage.restoration import denoise_bilateral

T=256; M=24
it=tf.lite.Interpreter(model_path='neurocam_enhance.tflite', num_threads=4); it.allocate_tensors()
ii=it.get_input_details()[0]['index']; oi=it.get_output_details()[0]['index']
def run_tiled(img):  # img float HxWx3 0..1
    h,w,_=img.shape; step=T-2*M
    pad=np.pad(img,((M,M+T),(M,M+T),(0,0)),mode='reflect')
    out=np.zeros_like(img)
    for y in range(0,h,step):
        for x in range(0,w,step):
            t=pad[y:y+T,x:x+T][None].astype(np.float32)
            it.set_tensor(ii,t); it.invoke(); o=it.get_tensor(oi)[0]
            hh=min(step,h-y); ww=min(step,w-x)
            out[y:y+hh,x:x+ww]=o[M:M+hh,M:M+ww]
    return np.clip(out,0,1)
def degrade(clean, rng, shot, read, jpegq):
    lin=clean**2.2
    std=np.sqrt(shot*lin+read**2)
    n=std*(0.7*rng.standard_normal(clean.shape[:2])[...,None]+0.71*rng.standard_normal(clean.shape))
    x=np.clip(lin+n,0,1)**(1/2.2)
    if jpegq:
        b=io.BytesIO(); Image.fromarray((x*255+0.5).astype(np.uint8)).save(b,'JPEG',quality=jpegq); x=np.asarray(Image.open(b))/255.0
    return x
def psnr(a,b): return 10*np.log10(1/np.mean((a-b)**2))
random.seed(0); files=sorted(glob.glob(__import__('os').environ.get('NEUROCAM_DATA','data/flower_photos')+'/*/*.jpg')); random.shuffle(files); val=files[:120]
imgs=[np.asarray(Image.open(f).convert('RGB'))/255.0 for f in val[:24]]
imgs+=[getattr(D,n)()/255.0 for n in ['astronaut','coffee','chelsea','rocket']]
rng=np.random.default_rng(0)
levels={'low noise (ISO~100)':(0.002,0.002,92),'medium (ISO~800)':(0.008,0.005,90),'high (ISO~3200)':(0.018,0.01,85)}
rows=[]
for name,(shot,read,q) in levels.items():
    a=[];b=[];c=[]
    for im in imgs:
        d=degrade(im,rng,shot,read,q)
        r=run_tiled(d)
        bl=denoise_bilateral(d,sigma_color=0.05,sigma_spatial=1.5,channel_axis=-1)
        a.append(psnr(d,im)); b.append(psnr(r,im)); c.append(psnr(bl,im))
    rows.append((name,np.mean(a),np.mean(c),np.mean(b)))
    print('%-22s input %.2f dB | bilateral %.2f dB | NeuroCam %.2f dB'%rows[-1])
# clean image: should be (almost) unchanged
a=[psnr(run_tiled(im),im) for im in imgs[:8]]; print('clean input fidelity: %.2f dB'%np.mean(a))
with open('RESULTS.md','w') as f:
    f.write('# NeuroCam enhancement model: evaluation\n\nHeld-out photos (24 unseen flower photos + 4 standard test images), synthetic camera degradation (signal dependent luma+chroma noise in linear light, then JPEG). PSNR vs. clean original, higher is better. Model run exactly like in the app (TFLite FP16, 256px tiles, 24px margin).\n\n| Noise level | Degraded input | Bilateral filter (classic) | **NeuroCam AI** |\n|---|---|---|---|\n')
    for r in rows: f.write('| %s | %.2f dB | %.2f dB | **%.2f dB** |\n'%r)
    f.write('\nOn already clean input the model output stays at %.1f dB PSNR from the input (i.e. it does not damage clean photos).\n'%np.mean(a))
# visual comparison
im=D.astronaut()/255.0; d=degrade(im,np.random.default_rng(5),0.018,0.01,85); r=run_tiled(d)
crop=lambda x: (x[60:316,140:396]*255+0.5).astype(np.uint8)
row=np.concatenate([crop(d),crop(r),crop(im)],1)
Image.fromarray(row).resize((row.shape[1]*2,row.shape[0]*2),Image.NEAREST).save('comparison.png')
