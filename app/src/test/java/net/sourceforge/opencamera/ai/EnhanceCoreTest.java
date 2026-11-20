package net.sourceforge.opencamera.ai;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class EnhanceCoreTest {

    private static EnhanceCore.IntImage randomImage(int w, int h, long seed) {
        Random r = new Random(seed);
        EnhanceCore.IntImage img = new EnhanceCore.IntImage(w, h);
        for(int i=0;i<img.data.length;i++)
            img.data[i] = 0xff000000 | (r.nextInt(256) << 16) | (r.nextInt(256) << 8) | r.nextInt(256);
        return img;
    }

    private static class IdentityModel implements EnhanceCore.TileModel {
        final int tile;
        int calls;
        IdentityModel(int tile) { this.tile = tile; }
        @Override public int getTileSize() { return tile; }
        @Override public void run(float[] input, float[] output) {
            calls++;
            System.arraycopy(input, 0, output, 0, input.length);
        }
    }

    /** 5x5 box blur within the tile (clamped at the tile borders, which the margin must hide). */
    private static class BoxBlurModel implements EnhanceCore.TileModel {
        final int tile;
        BoxBlurModel(int tile) { this.tile = tile; }
        @Override public int getTileSize() { return tile; }
        @Override public void run(float[] in, float[] out) {
            for(int y=0;y<tile;y++) {
                for(int x=0;x<tile;x++) {
                    for(int c=0;c<3;c++) {
                        float sum = 0;
                        for(int dy=-2;dy<=2;dy++) {
                            for(int dx=-2;dx<=2;dx++) {
                                int yy = Math.max(0, Math.min(tile-1, y+dy));
                                int xx = Math.max(0, Math.min(tile-1, x+dx));
                                sum += in[(yy*tile+xx)*3+c];
                            }
                        }
                        out[(y*tile+x)*3+c] = sum/25.0f;
                    }
                }
            }
        }
    }

    private static int reflect(int i, int n) {
        if( n == 1 ) return 0;
        int period = 2*(n-1);
        i = i % period;
        if( i < 0 ) i += period;
        return i < n ? i : period - i;
    }

    private static int[] referenceBoxBlur(EnhanceCore.IntImage img) {
        int[] res = new int[img.data.length];
        for(int y=0;y<img.height;y++) {
            for(int x=0;x<img.width;x++) {
                float[] sum = new float[3];
                for(int dy=-2;dy<=2;dy++) {
                    for(int dx=-2;dx<=2;dx++) {
                        int p = img.data[reflect(y+dy, img.height)*img.width + reflect(x+dx, img.width)];
                        sum[0] += ((p>>16)&0xff)/255.0f;
                        sum[1] += ((p>>8)&0xff)/255.0f;
                        sum[2] += (p&0xff)/255.0f;
                    }
                }
                res[y*img.width+x] = 0xff000000 | (EnhanceCore.toByte(sum[0]/25.0f) << 16) | (EnhanceCore.toByte(sum[1]/25.0f) << 8) | EnhanceCore.toByte(sum[2]/25.0f);
            }
        }
        return res;
    }

    @Test
    public void identityModelReproducesImage() {
        int[][] sizes = {{1,1},{7,3},{64,64},{65,130},{200,37},{301,257}};
        for(int[] sz : sizes) {
            EnhanceCore.IntImage src = randomImage(sz[0], sz[1], sz[0]*31L + sz[1]);
            EnhanceCore.IntImage dst = new EnhanceCore.IntImage(sz[0], sz[1]);
            IdentityModel model = new IdentityModel(64);
            assertTrue(EnhanceCore.runTiled(src, dst, model, 8, 1.0f, null));
            assertEquals(EnhanceCore.countTiles(sz[0], sz[1], 64, 8), model.calls);
            for(int i=0;i<src.data.length;i++)
                assertEquals("size " + sz[0] + "x" + sz[1] + " pixel " + i, src.data[i], dst.data[i]);
        }
    }

    @Test
    public void tilingIsSeamless() {
        // with margin >= the model's receptive radius, tiled result == whole image result
        EnhanceCore.IntImage src = randomImage(150, 97, 42);
        EnhanceCore.IntImage dst = new EnhanceCore.IntImage(150, 97);
        assertTrue(EnhanceCore.runTiled(src, dst, new BoxBlurModel(40), 4, 1.0f, null));
        int[] ref = referenceBoxBlur(src);
        for(int i=0;i<ref.length;i++) {
            int a = dst.data[i], b = ref[i];
            for(int s=0;s<24;s+=8) {
                int ca = (a >> s) & 0xff, cb = (b >> s) & 0xff;
                if( Math.abs(ca - cb) > 1 )
                    fail("pixel " + i + " differs: " + Integer.toHexString(a) + " vs " + Integer.toHexString(b));
            }
        }
    }

    @Test
    public void strengthBlends() {
        EnhanceCore.IntImage src = new EnhanceCore.IntImage(10, 10);
        java.util.Arrays.fill(src.data, 0xff646464); // 100
        EnhanceCore.TileModel plus = new EnhanceCore.TileModel() {
            @Override public int getTileSize() { return 16; }
            @Override public void run(float[] input, float[] output) {
                for(int i=0;i<input.length;i++)
                    output[i] = input[i] + 100.0f/255.0f;
            }
        };
        EnhanceCore.IntImage dst = new EnhanceCore.IntImage(10, 10);
        EnhanceCore.runTiled(src, dst, plus, 2, 0.5f, null);
        assertEquals(150, dst.data[55] & 0xff);
        EnhanceCore.runTiled(src, dst, plus, 2, 0.0f, null);
        assertEquals(100, dst.data[55] & 0xff);
        EnhanceCore.runTiled(src, dst, plus, 2, 1.0f, null);
        assertEquals(200, dst.data[55] & 0xff);
        assertEquals(0xff000000, dst.data[55] & 0xff000000);
    }

    @Test
    public void cancelStopsProcessing() {
        EnhanceCore.IntImage src = randomImage(100, 100, 1);
        EnhanceCore.IntImage dst = new EnhanceCore.IntImage(100, 100);
        IdentityModel model = new IdentityModel(32);
        assertFalse(EnhanceCore.runTiled(src, dst, model, 4, 1.0f, (done, total) -> done < 3));
        assertEquals(3, model.calls);
    }

    @Test(expected = IllegalArgumentException.class)
    public void inPlaceRejected() {
        EnhanceCore.IntImage src = randomImage(10, 10, 1);
        EnhanceCore.runTiled(src, src, new IdentityModel(16), 2, 1.0f, null);
    }

    @Test
    public void toneCurveIsMonotonicAndBounded() {
        for(EnhanceCore.Scene scene : EnhanceCore.Scene.values()) {
            long[] hist = new long[256];
            for(int i=30;i<220;i++)
                hist[i] = 100;
            EnhanceCore.ToneMapper m = EnhanceCore.buildToneMapper(hist, EnhanceCore.ToneParams.forScene(scene, 1.0f));
            int[] lut = m.getLut();
            for(int i=1;i<256;i++)
                assertTrue(scene + " monotonic at " + i, lut[i] >= lut[i-1]);
            assertTrue(lut[0] >= 0 && lut[255] <= 255);
            // stretch should darken the black point and brighten the white point (or leave unchanged)
            assertTrue(scene + " black", lut[30] <= 30 + 25);
            assertTrue(scene + " white", lut[219] >= 219 - 10);
        }
    }

    @Test
    public void zeroParamsGiveIdentity() {
        EnhanceCore.ToneParams p = EnhanceCore.ToneParams.forScene(EnhanceCore.Scene.GENERAL, 0.0f);
        long[] hist = new long[256];
        hist[100] = 10;
        EnhanceCore.ToneMapper m = EnhanceCore.buildToneMapper(hist, p);
        int[] px = {0xff123456, 0xff00ff00, 0xffffffff, 0xff000000};
        int[] copy = px.clone();
        m.apply(px, 0, px.length);
        for(int i=0;i<px.length;i++)
            assertEquals(copy[i], px[i]);
    }

    @Test
    public void vibranceProtectsSkin() {
        long[] hist = new long[256];
        for(int i=0;i<256;i++) hist[i] = 1;
        EnhanceCore.ToneParams p = new EnhanceCore.ToneParams();
        p.levels = 0; p.contrast = 0; p.vibrance = 0.5f; p.skinProtect = 1.0f;
        EnhanceCore.ToneMapper m = EnhanceCore.buildToneMapper(hist, p);
        int skin = 0xffe0ac90;
        int[] px = {skin, 0xff6080a0};
        m.apply(px, 0, 2);
        assertEquals(skin, px[0]);
        assertTrue("non skin colour should be more saturated", px[1] != 0xff6080a0);
    }

    @Test
    public void sceneClassification() {
        assertEquals(EnhanceCore.Scene.NIGHT, EnhanceCore.classifyScene(null, null, 3200, 120, -1));
        assertEquals(EnhanceCore.Scene.NIGHT, EnhanceCore.classifyScene(null, null, -1, 20, -1));
        assertEquals(EnhanceCore.Scene.PORTRAIT, EnhanceCore.classifyScene(null, null, 100, 120, 2));
        assertEquals(EnhanceCore.Scene.FOOD, EnhanceCore.classifyScene(new String[]{"Food", "Table"}, new float[]{0.9f, 0.6f}, 100, 120, -1));
        assertEquals(EnhanceCore.Scene.LANDSCAPE, EnhanceCore.classifyScene(new String[]{"Sky", "Cloud"}, new float[]{0.8f, 0.7f}, 100, 120, -1));
        assertEquals(EnhanceCore.Scene.NATURE, EnhanceCore.classifyScene(new String[]{"Flower", "Petal"}, new float[]{0.8f, 0.7f}, 100, 120, -1));
        assertEquals(EnhanceCore.Scene.DOCUMENT, EnhanceCore.classifyScene(new String[]{"Paper", "Text"}, new float[]{0.6f, 0.5f}, 100, 150, -1));
        assertEquals(EnhanceCore.Scene.GENERAL, EnhanceCore.classifyScene(new String[]{"Sky"}, new float[]{0.41f}, 100, 120, -1));
        assertEquals(EnhanceCore.Scene.GENERAL, EnhanceCore.classifyScene(new String[0], new float[0], 100, 120, -1));
    }

    @Test
    public void histogramAndMean() {
        EnhanceCore.IntImage img = new EnhanceCore.IntImage(4, 4);
        java.util.Arrays.fill(img.data, 0xff808080);
        long[] h = EnhanceCore.luminanceHistogram(img, 1);
        assertEquals(16, h[128]);
        assertEquals(128.0f, EnhanceCore.meanLuminance(h), 0.01f);
    }
    /** Nearest neighbour x4 "super resolution" model. */
    private static class NearestX4Model implements EnhanceCore.TileModel {
        final int tile;
        int calls;
        NearestX4Model(int tile) { this.tile = tile; }
        @Override public int getTileSize() { return tile; }
        @Override public void run(float[] in, float[] out) {
            calls++;
            int ot = tile*4;
            for(int y=0;y<ot;y++)
                for(int x=0;x<ot;x++)
                    for(int c=0;c<3;c++)
                        out[(y*ot+x)*3+c] = in[((y/4)*tile+(x/4))*3+c];
        }
    }

    @Test
    public void superResolutionTilingWithDownscale() {
        int[][] sizes = {{1,1},{9,5},{40,40},{77,53}};
        for(int[] sz : sizes) {
            EnhanceCore.IntImage lr = randomImage(sz[0], sz[1], sz[0]*7L + sz[1]);
            EnhanceCore.IntImage dst = new EnhanceCore.IntImage(sz[0]*2, sz[1]*2);
            NearestX4Model model = new NearestX4Model(24);
            assertTrue(EnhanceCore.runTiledSR(lr, dst, model, 4, 2, 4, null, 1.0f, null));
            assertEquals(EnhanceCore.countTiles(sz[0], sz[1], 24, 4), model.calls);
            for(int y=0;y<dst.height;y++)
                for(int x=0;x<dst.width;x++)
                    assertEquals("size " + sz[0] + "x" + sz[1] + " at " + x + "," + y, lr.data[(y/2)*lr.width + x/2], dst.data[y*dst.width + x]);
        }
    }

    @Test
    public void superResolutionBlendsWithOriginal() {
        EnhanceCore.IntImage lr = new EnhanceCore.IntImage(8, 8);
        java.util.Arrays.fill(lr.data, 0xffc8c8c8); // 200
        EnhanceCore.IntImage orig = new EnhanceCore.IntImage(16, 16);
        java.util.Arrays.fill(orig.data, 0xff646464); // 100
        EnhanceCore.IntImage dst = new EnhanceCore.IntImage(16, 16);
        EnhanceCore.runTiledSR(lr, dst, new NearestX4Model(8), 4, 2, 2, orig, 0.5f, null);
        assertEquals(150, dst.data[37] & 0xff);
    }

    @Test(expected = IllegalArgumentException.class)
    public void superResolutionRejectsWrongSize() {
        EnhanceCore.runTiledSR(new EnhanceCore.IntImage(8, 8), new EnhanceCore.IntImage(15, 16), new NearestX4Model(8), 4, 2, 2, null, 1.0f, null);
    }

    @Test
    public void whiteBalanceRemovesWarmCast() {
        // tungsten-like orange cast on a grey scene
        EnhanceCore.IntImage img = new EnhanceCore.IntImage(64, 64);
        Random r = new Random(3);
        for(int i=0;i<img.data.length;i++) {
            int v = 60 + r.nextInt(120);
            int red = Math.min(255, (int)(v*1.35)), green = v, blue = (int)(v*0.7);
            img.data[i] = 0xff000000 | (red << 16) | (green << 8) | blue;
        }
        float[] full = EnhanceCore.estimateWhiteBalanceGains(img, 1, 1.0f);
        assertTrue("red should be reduced", full[0] < 1.0f);
        assertTrue("blue should be boosted", full[2] > 1.0f);
        float[] half = EnhanceCore.estimateWhiteBalanceGains(img, 1, 0.5f);
        assertTrue("partial correction is weaker", half[2] < full[2] && half[0] > full[0]);
        float[] none = EnhanceCore.estimateWhiteBalanceGains(img, 1, 0.0f);
        assertEquals(1.0f, none[0], 0.0f);
        // apply partial correction: the cast should be reduced but not inverted
        long[] hist = EnhanceCore.luminanceHistogram(img, 1);
        EnhanceCore.ToneParams p = EnhanceCore.ToneParams.forScene(EnhanceCore.Scene.GENERAL, 0.0f);
        EnhanceCore.ToneMapper m = EnhanceCore.buildToneMapper(hist, p, half);
        int[] px = {0xff000000 | (135 << 16) | (100 << 8) | 70};
        m.apply(px, 0, 1);
        int rr = (px[0] >> 16) & 0xff, bb = px[0] & 0xff;
        assertTrue("cast reduced: " + rr + " " + bb, rr - bb < 135 - 70);
        assertTrue("still warm: " + rr + " " + bb, rr > bb);
    }

    @Test
    public void whiteBalanceNeutralImageUnchanged() {
        EnhanceCore.IntImage img = randomImage(50, 50, 9);
        for(int i=0;i<img.data.length;i++) {
            int v = img.data[i] & 0xff;
            img.data[i] = 0xff000000 | (v << 16) | (v << 8) | v;
        }
        float[] g = EnhanceCore.estimateWhiteBalanceGains(img, 1, 1.0f);
        for(float v : g)
            assertEquals(1.0f, v, 0.01f);
    }
}
