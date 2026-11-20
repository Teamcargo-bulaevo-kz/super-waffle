package net.sourceforge.opencamera;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import android.view.View;

import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.rules.ActivityScenarioRule;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.rule.GrantPermissionRule;

import net.sourceforge.opencamera.ai.EnhanceActivity;
import net.sourceforge.opencamera.ai.EnhanceCore;
import net.sourceforge.opencamera.ai.NeuroEnhancer;

import org.junit.FixMethodOrder;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.Random;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/** On-device tests for the NeuroCam AI pipeline. Run with:
 *  ./gradlew connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=net.sourceforge.opencamera.NeuroCamInstrumentedTest
 */
@RunWith(AndroidJUnit4.class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class NeuroCamInstrumentedTest {
    private static final String TAG = "NeuroCamTest";

    static final Intent intent;
    static {
        intent = new Intent(ApplicationProvider.getApplicationContext(), MainActivity.class);
        TestUtils.setDefaultIntent(intent);
        intent.putExtra("test_project_junit4", true);
        TestUtils.initTest(ApplicationProvider.getApplicationContext());
    }

    @Rule
    public final GrantPermissionRule permissionRule = GrantPermissionRule.grant(
            Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO);

    @Rule
    public final ActivityScenarioRule<MainActivity> mActivityRule = new ActivityScenarioRule<>(intent);

    private Bitmap loadTestImage() throws Exception {
        try( InputStream in = InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("ai_test_astronaut.png") ) {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inMutable = true;
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            return BitmapFactory.decodeStream(in, null, options);
        }
    }

    /** Adds deterministic, camera like (luma + chroma, signal dependent) noise. */
    private static Bitmap addNoise(Bitmap clean, float sigma) {
        int w = clean.getWidth(), h = clean.getHeight();
        int[] px = new int[w*h];
        clean.getPixels(px, 0, w, 0, 0, w, h);
        Random rnd = new Random(1234);
        for(int i=0;i<px.length;i++) {
            int p = px[i];
            double luma = rnd.nextGaussian();
            int[] c = {(p >> 16) & 0xff, (p >> 8) & 0xff, p & 0xff};
            for(int k=0;k<3;k++) {
                double n = sigma * (0.7*luma + 0.71*rnd.nextGaussian());
                c[k] = Math.max(0, Math.min(255, (int)Math.round(c[k] + n)));
            }
            px[i] = 0xff000000 | (c[0] << 16) | (c[1] << 8) | c[2];
        }
        Bitmap out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        out.setPixels(px, 0, w, 0, 0, w, h);
        return out;
    }

    private static double psnr(Bitmap a, Bitmap b) {
        int w = a.getWidth(), h = a.getHeight();
        int[] pa = new int[w*h], pb = new int[w*h];
        a.getPixels(pa, 0, w, 0, 0, w, h);
        b.getPixels(pb, 0, w, 0, 0, w, h);
        double se = 0;
        for(int i=0;i<pa.length;i++) {
            for(int s=0;s<24;s+=8) {
                double d = ((pa[i] >> s) & 0xff) - ((pb[i] >> s) & 0xff);
                se += d*d;
            }
        }
        double mse = se / (3.0*pa.length);
        return 10.0*Math.log10(255.0*255.0/mse);
    }

    private void checkDenoise(boolean gpu) throws Exception {
        Bitmap clean = loadTestImage();
        assertNotNull(clean);
        Bitmap noisy = addNoise(clean, 12.0f);
        NeuroEnhancer enhancer = NeuroEnhancer.getInstance(ApplicationProvider.getApplicationContext());
        long t0 = System.currentTimeMillis();
        Bitmap restored = enhancer.neuralOnly(noisy, gpu);
        long time = System.currentTimeMillis() - t0;
        double before = psnr(noisy, clean);
        double after = psnr(restored, clean);
        Log.i(TAG, "AI_RESULT gpu_requested=" + gpu + " gpu_used=" + enhancer.isUsingGpu() + " psnr noisy=" + before + " restored=" + after + " time=" + time + "ms");
        assertEquals(clean.getWidth(), restored.getWidth());
        assertEquals(clean.getHeight(), restored.getHeight());
        assertTrue("neural network should improve PSNR by at least 3dB: " + before + " -> " + after, after > before + 3.0);
    }

    @Test
    public void testA_NeuralDenoiseCpu() throws Exception {
        checkDenoise(false);
    }

    @Test
    public void testB_NeuralDenoiseGpu() throws Exception {
        // falls back to the CPU if the emulator / device has no supported GPU
        checkDenoise(true);
    }

    @Test
    public void testC_FullPipeline() throws Exception {
        Bitmap clean = loadTestImage();
        NeuroEnhancer enhancer = NeuroEnhancer.getInstance(ApplicationProvider.getApplicationContext());
        NeuroEnhancer.Result result = enhancer.enhance(addNoise(clean, 8.0f), 800, 0.8f, true, true);
        Log.i(TAG, "AI_RESULT full pipeline scene=" + result.scene + " neural=" + result.neuralApplied + " gpu=" + result.usedGpu + " time=" + result.timeMs + "ms error=" + result.error);
        assertNull(result.error);
        assertTrue(result.neuralApplied);
        assertNotNull(result.bitmap);
        assertEquals(clean.getWidth(), result.bitmap.getWidth());
        // a photo of a person: the scene detector should not think it's food/document/night
        assertTrue("unexpected scene " + result.scene,
                result.scene != EnhanceCore.Scene.FOOD && result.scene != EnhanceCore.Scene.DOCUMENT && result.scene != EnhanceCore.Scene.NIGHT);
    }

    private <T> T getActivityValue(java.util.function.Function<MainActivity, T> cb) {
        AtomicReference<T> resultRef = new AtomicReference<>();
        mActivityRule.getScenario().onActivity(activity -> resultRef.set(cb.apply(activity)));
        return resultRef.get();
    }

    @Test
    public void testD_TakePhotoWithAI() throws Exception {
        long start = System.currentTimeMillis();
        while( !getActivityValue(a -> a.getPreview().isPreviewStarted()) ) {
            assertTrue("camera preview did not start", System.currentTimeMillis() - start < 30000);
            Thread.sleep(200);
        }
        Thread.sleep(1000);
        NeuroEnhancer.lastResult = null;
        mActivityRule.getScenario().onActivity(activity -> {
            View takePhotoButton = activity.findViewById(R.id.take_photo);
            takePhotoButton.performClick();
        });
        start = System.currentTimeMillis();
        while( NeuroEnhancer.lastResult == null || getActivityValue(a -> a.getApplicationInterface().getImageSaver().getNImagesToSave()) > 0 ) {
            assertTrue("photo was not AI processed and saved in time", System.currentTimeMillis() - start < 120000);
            Thread.sleep(250);
        }
        NeuroEnhancer.Result result = NeuroEnhancer.lastResult;
        Log.i(TAG, "AI_RESULT camera photo " + result.bitmap.getWidth() + "x" + result.bitmap.getHeight() + " scene=" + result.scene + " gpu=" + result.usedGpu + " time=" + result.timeMs + "ms error=" + result.error);
        assertTrue(result.neuralApplied);
        assertNull(result.error);
        assertTrue(result.bitmap.getWidth() > 0);
    }
    @Test
    public void testE_GenerativeDetail() throws Exception {
        Bitmap clean = loadTestImage();
        int w = clean.getWidth(), h = clean.getHeight();
        // soft, slightly noisy version: half resolution information only
        Bitmap soft = Bitmap.createScaledBitmap(Bitmap.createScaledBitmap(clean, w/2, h/2, true), w, h, true);
        soft = addNoise(soft, 4.0f);
        NeuroEnhancer enhancer = NeuroEnhancer.getInstance(ApplicationProvider.getApplicationContext());
        long t0 = System.currentTimeMillis();
        Bitmap detail = enhancer.detailOnly(soft, true);
        long time = System.currentTimeMillis() - t0;
        double before = psnr(soft, clean);
        double after = psnr(detail, clean);
        Log.i(TAG, "AI_RESULT detail psnr soft=" + before + " detail=" + after + " time=" + time + "ms");
        assertEquals(w, detail.getWidth());
        assertEquals(h, detail.getHeight());
        assertTrue("AI detail should restore the soft image: " + before + " -> " + after, after > before + 0.4);
    }

    @Test
    public void testF_EnhanceActivityFromGallery() throws Exception {
        // put a photo in the gallery, then "share" it to NeuroCam with AI detail on
        android.content.Context context = ApplicationProvider.getApplicationContext();
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, "neurocam_test_" + System.currentTimeMillis() + ".png");
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/NeuroCamTest");
        Uri uri = context.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        assertNotNull(uri);
        try( OutputStream out = context.getContentResolver().openOutputStream(uri);
             InputStream in = InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("ai_test_astronaut.png") ) {
            byte[] buf = new byte[65536];
            int n;
            while( (n = in.read(buf)) > 0 )
                out.write(buf, 0, n);
        }
        EnhanceActivity.lastSavedUri = null;
        EnhanceActivity.lastResult = null;
        Intent intent = new Intent(Intent.ACTION_SEND);
        intent.setClassName(context, EnhanceActivity.class.getName());
        intent.setType("image/png");
        intent.putExtra(Intent.EXTRA_STREAM, uri);
        intent.putExtra(EnhanceActivity.EXTRA_AUTO_START, true);
        intent.putExtra(EnhanceActivity.EXTRA_DETAIL, true);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_GRANT_READ_URI_PERMISSION);
        context.startActivity(intent);
        long start = System.currentTimeMillis();
        while( EnhanceActivity.lastResult == null ) {
            assertTrue("gallery enhancement did not finish", System.currentTimeMillis() - start < 300000);
            Thread.sleep(500);
        }
        NeuroEnhancer.Result result = EnhanceActivity.lastResult;
        Log.i(TAG, "AI_RESULT gallery enhance scene=" + result.scene + " detail=" + result.detailApplied + " gpu=" + result.usedGpu + " time=" + result.timeMs + "ms saved=" + EnhanceActivity.lastSavedUri + " error=" + result.error);
        assertTrue(result.detailApplied);
        assertNull(result.error);
        assertNotNull(EnhanceActivity.lastSavedUri);
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        try( InputStream in = context.getContentResolver().openInputStream(EnhanceActivity.lastSavedUri) ) {
            BitmapFactory.decodeStream(in, null, o);
        }
        assertEquals(512, o.outWidth);
        assertEquals(512, o.outHeight);
    }
}
