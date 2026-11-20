package net.sourceforge.opencamera.ai;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.preference.PreferenceManager;
import android.util.Log;

import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.label.ImageLabel;
import com.google.mlkit.vision.label.ImageLabeler;
import com.google.mlkit.vision.label.ImageLabeling;
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * NeuroCam AI photo pipeline (Android side):
 * <ol>
 *     <li>Scene recognition with an on-device ML Kit image labelling model, combined with ISO and
 *     brightness, picks a scene profile (portrait, food, landscape, nature, night, document...).</li>
 *     <li>Either
 *     <ul>
 *         <li><b>restoration</b>: a convolutional network trained for this app
 *         (assets/neurocam_enhance.tflite) removes sensor noise, JPEG artefacts and softness at the
 *         original resolution, or</li>
 *         <li><b>AI detail</b> (generative): Real-ESRGAN general x4v3 (assets/realesr_general_x4v3.tflite)
 *         re-synthesises fine texture: the photo is downscaled 2x, super-resolved 4x and
 *         downscaled 2x again on the fly, giving a clean, crisp result at the original size.</li>
 *     </ul></li>
 *     <li>Scene adaptive white balance (removes part of e.g. the yellow cast of lamps), tone curve,
 *     auto levels and vibrance (with skin tone protection).</li>
 * </ol>
 * Networks run tile by tile on the GPU, with automatic CPU fallback. All work is done on a single
 * dedicated thread, since the TFLite GPU delegate must be used from the thread that created it.
 */
public class NeuroEnhancer {
    private static final String TAG = "NeuroEnhancer";

    public static final String PREF_ENABLED = "preference_ai_enhance";
    public static final String PREF_STRENGTH = "preference_ai_strength";
    public static final String PREF_SCENE = "preference_ai_scene";
    public static final String PREF_GPU = "preference_ai_gpu";
    public static final String PREF_DETAIL = "preference_ai_detail";
    public static final String PREF_WHITE_BALANCE = "preference_ai_white_balance";

    static final String RESTORE_MODEL_ASSET = "neurocam_enhance.tflite";
    static final String DETAIL_MODEL_ASSET = "realesr_general_x4v3.tflite";
    /** Context margin for each tile of the restoration net; must cover its receptive field (~35 px). */
    static final int RESTORE_MARGIN = 24;
    /** Context margin (in low resolution pixels) for the Real-ESRGAN tiles (receptive field ~69 px). */
    static final int DETAIL_MARGIN = 24;

    private static NeuroEnhancer instance;

    /** Most recent result, for tests. */
    public static volatile Result lastResult;

    private final Context context;
    private final ExecutorService executor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "NeuroEnhancer");
        t.setPriority(Thread.NORM_PRIORITY);
        return t;
    });

    // only accessed on the executor thread:
    private final TfliteModel restoreModel;
    private final TfliteModel detailModel;
    private ImageLabeler labeler;

    /** Processing options. */
    public static class Options {
        public float strength = 0.8f;
        public boolean scene = true;
        public boolean gpu = true;
        public boolean detail = false;
        public boolean whiteBalance = true;
        /** Optional progress callback (called on the processing thread), 0..1. */
        public ProgressCallback progress;

        public static Options fromPreferences(Context context) {
            SharedPreferences sp = PreferenceManager.getDefaultSharedPreferences(context);
            Options o = new Options();
            o.strength = getStrength(context);
            o.scene = sp.getBoolean(PREF_SCENE, true);
            o.gpu = sp.getBoolean(PREF_GPU, true);
            o.detail = sp.getBoolean(PREF_DETAIL, false);
            o.whiteBalance = sp.getBoolean(PREF_WHITE_BALANCE, true);
            return o;
        }
    }

    public interface ProgressCallback {
        void onProgress(float fraction);
    }

    /** Result information, for logging and tests. */
    public static class Result {
        public Bitmap bitmap;
        public EnhanceCore.Scene scene = EnhanceCore.Scene.GENERAL;
        public boolean usedGpu;
        public boolean neuralApplied;
        public boolean detailApplied;
        public float[] whiteBalanceGains = {1.0f, 1.0f, 1.0f};
        public long timeMs;
        public String error;
    }

    public static synchronized NeuroEnhancer getInstance(Context context) {
        if( instance == null )
            instance = new NeuroEnhancer(context.getApplicationContext());
        return instance;
    }

    private NeuroEnhancer(Context context) {
        this.context = context;
        this.restoreModel = new TfliteModel(context, RESTORE_MODEL_ASSET);
        this.detailModel = new TfliteModel(context, DETAIL_MODEL_ASSET);
    }

    public static boolean isEnabled(Context context) {
        SharedPreferences sp = PreferenceManager.getDefaultSharedPreferences(context);
        return sp.getBoolean(PREF_ENABLED, true);
    }

    public static float getStrength(Context context) {
        SharedPreferences sp = PreferenceManager.getDefaultSharedPreferences(context);
        String s = sp.getString(PREF_STRENGTH, "medium");
        switch( s ) {
            case "low":
                return 0.5f;
            case "high":
                return 1.0f;
            case "medium":
            default:
                return 0.8f;
        }
    }

    /**
     * Enhances the photo using the user's preferences. Blocks until complete.
     * @param bitmap the source photo (already rotated to the correct orientation). Not recycled.
     * @param iso capture ISO, or -1 if not known.
     * @return the result; result.bitmap is the enhanced bitmap, or the source bitmap if processing failed.
     */
    public Result enhance(final Bitmap bitmap, final int iso) {
        return enhance(bitmap, iso, Options.fromPreferences(context));
    }

    public Result enhance(final Bitmap bitmap, final int iso, final float strength, final boolean useScene, final boolean useGpu) {
        Options o = new Options();
        o.strength = strength;
        o.scene = useScene;
        o.gpu = useGpu;
        o.detail = false;
        return enhance(bitmap, iso, o);
    }

    public Result enhance(final Bitmap bitmap, final int iso, final Options options) {
        Future<Result> future = executor.submit(() -> enhanceOnThread(bitmap, iso, options));
        try {
            return future.get(30, TimeUnit.MINUTES);
        }
        catch(Exception e) {
            Log.e(TAG, "enhance failed", e);
            future.cancel(true);
            Result r = new Result();
            r.bitmap = bitmap;
            r.error = e.toString();
            return r;
        }
    }

    private Result enhanceOnThread(Bitmap src, int iso, Options options) {
        long t0 = SystemClock.elapsedRealtime();
        Result result = new Result();
        result.bitmap = src;
        BitmapImage srcImage = new BitmapImage(src);
        final ProgressCallback progress = options.progress;
        if( progress != null )
            progress.onProgress(0.0f);

        // 1. scene analysis
        long[] hist = EnhanceCore.luminanceHistogram(srcImage, Math.max(1, Math.max(src.getWidth(), src.getHeight()) / 512));
        float meanLuma = EnhanceCore.meanLuminance(hist);
        String[] labels = null;
        float[] confidences = null;
        if( options.scene ) {
            try {
                List<ImageLabel> imageLabels = labelImage(src);
                labels = new String[imageLabels.size()];
                confidences = new float[imageLabels.size()];
                for(int i=0;i<imageLabels.size();i++) {
                    labels[i] = imageLabels.get(i).getText();
                    confidences[i] = imageLabels.get(i).getConfidence();
                }
                if( Log.isLoggable(TAG, Log.DEBUG) ) {
                    for(int i=0;i<labels.length;i++)
                        Log.d(TAG, "label: " + labels[i] + " " + confidences[i]);
                }
            }
            catch(Throwable e) {
                Log.e(TAG, "scene labelling failed", e);
            }
        }
        result.scene = EnhanceCore.classifyScene(labels, confidences, iso, meanLuma, -1);
        Log.i(TAG, "scene: " + result.scene + " iso " + iso + " mean luma " + meanLuma + " detail " + options.detail);
        if( progress != null )
            progress.onProgress(0.05f);

        // 2. neural stage
        Bitmap out = null;
        EnhanceCore.ProgressListener tileProgress = progress == null ? null : (done, total) -> {
            progress.onProgress(0.05f + 0.9f * done / total);
            return true;
        };
        if( options.detail ) {
            try {
                out = runDetail(src, options.gpu, options.strength, tileProgress);
                result.neuralApplied = true;
                result.detailApplied = true;
                result.usedGpu = detailModel.usesGpu;
            }
            catch(Throwable e) {
                Log.e(TAG, "AI detail failed, falling back to restoration", e);
                result.error = e.toString();
                out = null;
            }
        }
        if( out == null ) {
            try {
                out = runRestore(src, options.gpu, neuralStrengthFor(result.scene, options.strength), tileProgress);
                result.neuralApplied = true;
                result.usedGpu = restoreModel.usesGpu;
                if( !options.detail )
                    result.error = null;
            }
            catch(Throwable e) {
                Log.e(TAG, "neural enhancement failed, continuing with tone only", e);
                result.error = e.toString();
                out = null;
            }
        }
        if( out == null ) {
            out = src.copy(Bitmap.Config.ARGB_8888, true);
            if( out == null ) {
                result.timeMs = SystemClock.elapsedRealtime() - t0;
                return result;
            }
        }

        // 3. white balance, tone & colour
        BitmapImage outImage = new BitmapImage(out);
        int sampleStep = Math.max(1, Math.max(out.getWidth(), out.getHeight()) / 1024);
        long[] outHist = EnhanceCore.luminanceHistogram(outImage, sampleStep);
        EnhanceCore.ToneParams toneParams = EnhanceCore.ToneParams.forScene(result.scene, options.strength);
        float[] gains = options.whiteBalance ?
                EnhanceCore.estimateWhiteBalanceGains(outImage, sampleStep, toneParams.whiteBalance) :
                new float[]{1.0f, 1.0f, 1.0f};
        result.whiteBalanceGains = gains;
        EnhanceCore.ToneMapper mapper = EnhanceCore.buildToneMapper(outHist, toneParams, gains);
        EnhanceCore.applyTone(outImage, mapper);

        result.bitmap = out;
        result.timeMs = SystemClock.elapsedRealtime() - t0;
        lastResult = result;
        if( progress != null )
            progress.onProgress(1.0f);
        Log.i(TAG, "enhanced " + src.getWidth() + "x" + src.getHeight() + " in " + result.timeMs + "ms, gpu: " + result.usedGpu
                + " detail: " + result.detailApplied + " wb gains: " + gains[0] + "," + gains[1] + "," + gains[2]);
        return result;
    }

    private Bitmap runRestore(Bitmap src, boolean useGpu, float strength, EnhanceCore.ProgressListener listener) throws IOException {
        restoreModel.ensure(useGpu);
        Bitmap out = Bitmap.createBitmap(src.getWidth(), src.getHeight(), Bitmap.Config.ARGB_8888);
        try {
            EnhanceCore.runTiled(new BitmapImage(src), new BitmapImage(out), restoreModel, RESTORE_MARGIN, strength, listener);
        }
        catch(RuntimeException e) {
            out.recycle();
            restoreModel.onRunFailure();
            throw e;
        }
        return out;
    }

    /** Generative detail: downscale 2x, Real-ESRGAN 4x, fused 2x box downscale back to the original size. */
    private Bitmap runDetail(Bitmap src, boolean useGpu, float strength, EnhanceCore.ProgressListener listener) throws IOException {
        detailModel.ensure(useGpu);
        final int w = src.getWidth(), h = src.getHeight();
        final int lw = Math.max(1, w / 2), lh = Math.max(1, h / 2);
        Bitmap lr = Bitmap.createScaledBitmap(src, lw, lh, true);
        Bitmap out = Bitmap.createBitmap(lw*2, lh*2, Bitmap.Config.ARGB_8888);
        try {
            Bitmap orig = src;
            if( w != lw*2 || h != lh*2 )
                orig = Bitmap.createBitmap(src, 0, 0, lw*2, lh*2); // odd sizes: crop by 1px for the blend
            EnhanceCore.runTiledSR(new BitmapImage(lr), new BitmapImage(out), detailModel, 4, 2, DETAIL_MARGIN,
                    new BitmapImage(orig), Math.min(1.0f, 0.25f + strength), listener);
            if( orig != src )
                orig.recycle();
        }
        catch(RuntimeException e) {
            out.recycle();
            detailModel.onRunFailure();
            throw e;
        }
        finally {
            if( lr != src )
                lr.recycle();
        }
        if( out.getWidth() != w || out.getHeight() != h ) {
            Bitmap scaled = Bitmap.createScaledBitmap(out, w, h, true);
            out.recycle();
            out = scaled;
        }
        return out;
    }

    /** Runs only the restoration network (full strength) - used by tests to measure quality. */
    public Bitmap neuralOnly(final Bitmap src, final boolean useGpu) throws Exception {
        return executor.submit(() -> runRestore(src, useGpu, 1.0f, null)).get(5, TimeUnit.MINUTES);
    }

    /** Runs only the generative detail stage (full strength) - used by tests. */
    public Bitmap detailOnly(final Bitmap src, final boolean useGpu) throws Exception {
        return executor.submit(() -> runDetail(src, useGpu, 1.0f, null)).get(10, TimeUnit.MINUTES);
    }

    /** Whether the restoration interpreter runs on the GPU (valid after an enhancement). */
    public boolean isUsingGpu() {
        try {
            return executor.submit(() -> restoreModel.usesGpu).get();
        }
        catch(Exception e) {
            return false;
        }
    }

    static float neuralStrengthFor(EnhanceCore.Scene scene, float strength) {
        switch( scene ) {
            case NIGHT:
                return Math.min(1.0f, strength * 1.2f); // most noise: full strength
            case PORTRAIT:
                return strength * 0.85f; // keep natural skin texture
            default:
                return strength;
        }
    }

    private List<ImageLabel> labelImage(Bitmap src) throws Exception {
        if( labeler == null ) {
            labeler = ImageLabeling.getClient(new ImageLabelerOptions.Builder().setConfidenceThreshold(0.4f).build());
        }
        int maxSide = Math.max(src.getWidth(), src.getHeight());
        Bitmap small = src;
        if( maxSide > 640 ) {
            float scale = 640.0f / maxSide;
            small = Bitmap.createScaledBitmap(src, Math.max(1, Math.round(src.getWidth()*scale)), Math.max(1, Math.round(src.getHeight()*scale)), true);
        }
        try {
            return Tasks.await(labeler.process(InputImage.fromBitmap(small, 0)), 20, TimeUnit.SECONDS);
        }
        finally {
            if( small != src )
                small.recycle();
        }
    }

    /** A TFLite model with fixed square input tile, run on GPU if possible, else CPU (XNNPACK). */
    static final class TfliteModel implements EnhanceCore.TileModel {
        private final Context context;
        private final String asset;
        private Interpreter interpreter;
        private GpuDelegate gpuDelegate;
        boolean usesGpu;
        private boolean gpuFailed;
        private boolean gpuUnsupported;
        private int tileSize;
        private ByteBuffer inputBuffer;
        private ByteBuffer outputBuffer;

        TfliteModel(Context context, String asset) {
            this.context = context;
            this.asset = asset;
        }

        void ensure(boolean useGpu) throws IOException {
            if( interpreter != null ) {
                if( usesGpu == useGpu )
                    return;
                if( !usesGpu && (gpuFailed || gpuUnsupported) )
                    return; // GPU requested but not usable: keep the CPU interpreter
            }
            close();
            MappedByteBuffer model = loadModel();
            if( useGpu && !gpuFailed ) {
                try {
                    CompatibilityList compatList = new CompatibilityList();
                    if( compatList.isDelegateSupportedOnThisDevice() ) {
                        GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                        delegateOptions.setPrecisionLossAllowed(true);
                        delegateOptions.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
                        gpuDelegate = new GpuDelegate(delegateOptions);
                        Interpreter.Options options = new Interpreter.Options();
                        options.addDelegate(gpuDelegate);
                        interpreter = new Interpreter(model, options);
                        usesGpu = true;
                    }
                    else {
                        Log.i(TAG, "GPU delegate not supported on this device");
                        gpuUnsupported = true;
                    }
                }
                catch(Throwable e) {
                    Log.e(TAG, "failed to create GPU interpreter for " + asset, e);
                    gpuFailed = true;
                    close();
                }
            }
            if( interpreter == null ) {
                Interpreter.Options options = new Interpreter.Options();
                options.setNumThreads(Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors())));
                options.setUseXNNPACK(true);
                interpreter = new Interpreter(model, options);
                usesGpu = false;
            }
            int[] inShape = interpreter.getInputTensor(0).shape();
            int[] outShape = interpreter.getOutputTensor(0).shape();
            tileSize = inShape[1];
            inputBuffer = ByteBuffer.allocateDirect(inShape[1]*inShape[2]*inShape[3]*4).order(ByteOrder.nativeOrder());
            outputBuffer = ByteBuffer.allocateDirect(outShape[1]*outShape[2]*outShape[3]*4).order(ByteOrder.nativeOrder());
            Log.i(TAG, "interpreter ready: " + asset + " tile " + tileSize + " gpu " + usesGpu);
        }

        void onRunFailure() {
            if( usesGpu ) {
                // the GPU delegate can fail on some devices at run time; fall back to the CPU next time
                gpuFailed = true;
                close();
            }
        }

        @Override
        public int getTileSize() {
            return tileSize;
        }

        @Override
        public void run(float[] input, float[] output) {
            inputBuffer.rewind();
            inputBuffer.asFloatBuffer().put(input);
            outputBuffer.rewind();
            interpreter.run(inputBuffer, outputBuffer);
            outputBuffer.rewind();
            outputBuffer.asFloatBuffer().get(output);
        }

        private MappedByteBuffer loadModel() throws IOException {
            try( AssetFileDescriptor fd = context.getAssets().openFd(asset);
                 FileInputStream stream = new FileInputStream(fd.getFileDescriptor()) ) {
                FileChannel channel = stream.getChannel();
                return channel.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
            }
        }

        void close() {
            if( interpreter != null ) {
                interpreter.close();
                interpreter = null;
            }
            if( gpuDelegate != null ) {
                gpuDelegate.close();
                gpuDelegate = null;
            }
            usesGpu = false;
        }
    }

    /** Wraps an android Bitmap as a PixelImage. */
    public static final class BitmapImage implements EnhanceCore.PixelImage {
        private final Bitmap bitmap;

        public BitmapImage(Bitmap bitmap) {
            this.bitmap = bitmap;
        }

        @Override public int getWidth() { return bitmap.getWidth(); }
        @Override public int getHeight() { return bitmap.getHeight(); }

        @Override
        public void getPixels(int[] pixels, int offset, int stride, int x, int y, int width, int height) {
            bitmap.getPixels(pixels, offset, stride, x, y, width, height);
        }

        @Override
        public void setPixels(int[] pixels, int offset, int stride, int x, int y, int width, int height) {
            bitmap.setPixels(pixels, offset, stride, x, y, width, height);
        }
    }
}
