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
import java.nio.FloatBuffer;
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
 *     <li>A convolutional neural network (assets/neurocam_enhance.tflite, trained for this app) removes
 *     sensor noise, JPEG artefacts and lens / ISP softness, run tile by tile over the full
 *     resolution photo on the GPU (with automatic CPU fallback).</li>
 *     <li>Scene adaptive tone curve, auto levels and vibrance (with skin tone protection).</li>
 * </ol>
 * All work is done on a single dedicated thread, since the TFLite GPU delegate must be used from
 * the thread that created it.
 */
public class NeuroEnhancer {
    private static final String TAG = "NeuroEnhancer";

    public static final String PREF_ENABLED = "preference_ai_enhance";
    public static final String PREF_STRENGTH = "preference_ai_strength";
    public static final String PREF_SCENE = "preference_ai_scene";
    public static final String PREF_GPU = "preference_ai_gpu";

    static final String MODEL_ASSET = "neurocam_enhance.tflite";
    /** Context margin for each tile; must cover the receptive field of the network (about 35 pixels). */
    static final int TILE_MARGIN = 24;

    private static NeuroEnhancer instance;

    private final Context context;
    private final ExecutorService executor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "NeuroEnhancer");
        t.setPriority(Thread.NORM_PRIORITY);
        return t;
    });

    // only accessed on the executor thread:
    private Interpreter interpreter;
    private GpuDelegate gpuDelegate;
    private boolean interpreterUsesGpu;
    private int tileSize;
    private ByteBuffer inputBuffer;
    private ByteBuffer outputBuffer;
    private ImageLabeler labeler;

    /** Result information, for logging and tests. */
    public static class Result {
        public Bitmap bitmap;
        public EnhanceCore.Scene scene = EnhanceCore.Scene.GENERAL;
        public boolean usedGpu;
        public boolean neuralApplied;
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
        SharedPreferences sp = PreferenceManager.getDefaultSharedPreferences(context);
        final float strength = getStrength(context);
        final boolean useScene = sp.getBoolean(PREF_SCENE, true);
        final boolean useGpu = sp.getBoolean(PREF_GPU, true);
        return enhance(bitmap, iso, strength, useScene, useGpu);
    }

    public Result enhance(final Bitmap bitmap, final int iso, final float strength, final boolean useScene, final boolean useGpu) {
        Future<Result> future = executor.submit(() -> enhanceOnThread(bitmap, iso, strength, useScene, useGpu));
        try {
            return future.get(10, TimeUnit.MINUTES);
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

    private Result enhanceOnThread(Bitmap src, int iso, float strength, boolean useScene, boolean useGpu) {
        long t0 = SystemClock.elapsedRealtime();
        Result result = new Result();
        result.bitmap = src;
        BitmapImage srcImage = new BitmapImage(src);

        // 1. scene analysis
        long[] hist = EnhanceCore.luminanceHistogram(srcImage, Math.max(1, Math.max(src.getWidth(), src.getHeight()) / 512));
        float meanLuma = EnhanceCore.meanLuminance(hist);
        String[] labels = null;
        float[] confidences = null;
        if( useScene ) {
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
        Log.i(TAG, "scene: " + result.scene + " iso " + iso + " mean luma " + meanLuma);

        // 2. neural restoration
        Bitmap out = null;
        try {
            ensureInterpreter(useGpu);
            out = Bitmap.createBitmap(src.getWidth(), src.getHeight(), Bitmap.Config.ARGB_8888);
            float neuralStrength = neuralStrengthFor(result.scene, strength);
            EnhanceCore.runTiled(srcImage, new BitmapImage(out), new EnhanceCore.TileModel() {
                @Override
                public int getTileSize() {
                    return tileSize;
                }

                @Override
                public void run(float[] input, float[] output) {
                    runModel(input, output);
                }
            }, TILE_MARGIN, neuralStrength, null);
            result.neuralApplied = true;
            result.usedGpu = interpreterUsesGpu;
        }
        catch(Throwable e) {
            Log.e(TAG, "neural enhancement failed, continuing with tone only", e);
            result.error = e.toString();
            if( out != null ) {
                out.recycle();
                out = null;
            }
            if( interpreterUsesGpu ) {
                // GPU delegate can fail on some devices at run time; fall back to CPU next time
                closeInterpreter();
            }
        }
        if( out == null ) {
            out = src.copy(Bitmap.Config.ARGB_8888, true);
            if( out == null ) {
                result.timeMs = SystemClock.elapsedRealtime() - t0;
                return result;
            }
        }

        // 3. tone & colour
        BitmapImage outImage = new BitmapImage(out);
        long[] outHist = EnhanceCore.luminanceHistogram(outImage, Math.max(1, Math.max(out.getWidth(), out.getHeight()) / 1024));
        EnhanceCore.ToneMapper mapper = EnhanceCore.buildToneMapper(outHist, EnhanceCore.ToneParams.forScene(result.scene, strength));
        EnhanceCore.applyTone(outImage, mapper);

        result.bitmap = out;
        result.timeMs = SystemClock.elapsedRealtime() - t0;
        Log.i(TAG, "enhanced " + src.getWidth() + "x" + src.getHeight() + " in " + result.timeMs + "ms, gpu: " + result.usedGpu);
        return result;
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

    private void ensureInterpreter(boolean useGpu) throws IOException {
        if( interpreter != null ) {
            if( interpreterUsesGpu == useGpu )
                return;
            if( !interpreterUsesGpu && (gpuFailed || gpuUnsupported) )
                return; // GPU requested but not usable: keep the CPU interpreter
        }
        closeInterpreter();
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
                    interpreterUsesGpu = true;
                }
                else {
                    Log.i(TAG, "GPU delegate not supported on this device");
                    gpuUnsupported = true;
                }
            }
            catch(Throwable e) {
                Log.e(TAG, "failed to create GPU interpreter", e);
                gpuFailed = true;
                closeInterpreter();
            }
        }
        if( interpreter == null ) {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors())));
            options.setUseXNNPACK(true);
            interpreter = new Interpreter(model, options);
            interpreterUsesGpu = false;
        }
        int[] shape = interpreter.getInputTensor(0).shape();
        tileSize = shape[1];
        int bytes = tileSize * tileSize * 3 * 4;
        inputBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder());
        outputBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder());
        Log.i(TAG, "interpreter ready, tile " + tileSize + " gpu " + interpreterUsesGpu);
    }

    private boolean gpuFailed;
    private boolean gpuUnsupported;

    private void runModel(float[] input, float[] output) {
        inputBuffer.rewind();
        FloatBuffer fin = inputBuffer.asFloatBuffer();
        fin.put(input);
        outputBuffer.rewind();
        try {
            interpreter.run(inputBuffer, outputBuffer);
        }
        catch(RuntimeException e) {
            if( interpreterUsesGpu )
                gpuFailed = true;
            throw e;
        }
        outputBuffer.rewind();
        outputBuffer.asFloatBuffer().get(output);
    }

    private MappedByteBuffer loadModel() throws IOException {
        try( AssetFileDescriptor fd = context.getAssets().openFd(MODEL_ASSET);
             FileInputStream stream = new FileInputStream(fd.getFileDescriptor()) ) {
            FileChannel channel = stream.getChannel();
            return channel.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
        }
    }

    private void closeInterpreter() {
        if( interpreter != null ) {
            interpreter.close();
            interpreter = null;
        }
        if( gpuDelegate != null ) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        interpreterUsesGpu = false;
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
