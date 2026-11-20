package net.sourceforge.opencamera.ai;

/**
 * Platform independent core of the NeuroCam AI pipeline.
 *
 * <p>Contains the tiled neural inference driver (so that full resolution photos can be processed
 * by a fixed-size network without seams) and the scene adaptive tone / colour stage.
 * It has no Android dependencies, so it is covered by plain JVM unit tests.</p>
 */
public final class EnhanceCore {

    private EnhanceCore() {
    }

    /** Abstraction of an ARGB_8888 image (an android Bitmap on device, an int[] in tests). */
    public interface PixelImage {
        int getWidth();
        int getHeight();
        void getPixels(int[] pixels, int offset, int stride, int x, int y, int width, int height);
        void setPixels(int[] pixels, int offset, int stride, int x, int y, int width, int height);
    }

    /** A neural network that maps a tile x tile RGB float image (NHWC, values 0..1) to an image of the same size. */
    public interface TileModel {
        int getTileSize();
        void run(float[] input, float[] output);
    }

    public interface ProgressListener {
        /** @return false to cancel processing. */
        boolean onProgress(int done, int total);
    }

    /** Simple int[] backed image. */
    public static final class IntImage implements PixelImage {
        public final int width, height;
        public final int[] data;

        public IntImage(int width, int height) {
            this(width, height, new int[width * height]);
        }

        public IntImage(int width, int height, int[] data) {
            this.width = width;
            this.height = height;
            this.data = data;
        }

        @Override public int getWidth() { return width; }
        @Override public int getHeight() { return height; }

        @Override
        public void getPixels(int[] pixels, int offset, int stride, int x, int y, int w, int h) {
            for(int j=0;j<h;j++) {
                System.arraycopy(data, (y+j)*width + x, pixels, offset + j*stride, w);
            }
        }

        @Override
        public void setPixels(int[] pixels, int offset, int stride, int x, int y, int w, int h) {
            for(int j=0;j<h;j++) {
                System.arraycopy(pixels, offset + j*stride, data, (y+j)*width + x, w);
            }
        }
    }

    /** Number of tiles that {@link #runTiled} will process for the given image size. */
    public static int countTiles(int width, int height, int tile, int margin) {
        int step = tile - 2*margin;
        if( step <= 0 )
            throw new IllegalArgumentException("margin too large for tile size");
        int nx = (width + step - 1) / step;
        int ny = (height + step - 1) / step;
        return nx * ny;
    }

    private static int reflect(int i, int n) {
        // mirror padding without repeating the edge pixel; handles any offset
        if( n == 1 )
            return 0;
        int period = 2*(n-1);
        i = i % period;
        if( i < 0 )
            i += period;
        return i < n ? i : period - i;
    }

    /**
     * Runs the model over the whole image using overlapping tiles. Each tile is read with
     * {@code margin} pixels of context on every side (mirror padded at the image borders) and only
     * the central part is written back, so as long as margin covers the receptive field of the
     * network the result is identical to running the network on the full image at once.
     *
     * <p>The result is blended with the original: out = in + strength * (net - in).</p>
     *
     * @param src source image (not modified)
     * @param dst destination image, same size as src; must be a different object from src, as
     *            the context margins of later tiles are read from src.
     * @return false if cancelled
     */
    public static boolean runTiled(PixelImage src, PixelImage dst, TileModel model, int margin, float strength, ProgressListener listener) {
        final int width = src.getWidth();
        final int height = src.getHeight();
        if( dst.getWidth() != width || dst.getHeight() != height )
            throw new IllegalArgumentException("size mismatch");
        if( src == dst )
            throw new IllegalArgumentException("in place processing not supported");
        final int tile = model.getTileSize();
        final int step = tile - 2*margin;
        if( step <= 0 )
            throw new IllegalArgumentException("margin too large for tile size");
        strength = Math.max(0.0f, Math.min(1.0f, strength));

        final int total = countTiles(width, height, tile, margin);
        final float[] in = new float[tile*tile*3];
        final float[] out = new float[tile*tile*3];
        final int[] outRow = new int[step];
        final float inv255 = 1.0f/255.0f;

        // the rows of the current band of tiles, including the context margin
        final int stripRows = tile;
        final int[] strip = new int[stripRows * width];
        final int[] rowBuf = new int[width];
        int done = 0;

        for(int ty=0;ty<height;ty+=step) {
            // load the source strip [ty - margin, ty - margin + tile) with mirror padding
            for(int r=0;r<stripRows;r++) {
                int sy = reflect(ty - margin + r, height);
                src.getPixels(rowBuf, 0, width, 0, sy, width, 1);
                System.arraycopy(rowBuf, 0, strip, r*width, width);
            }
            final int th = Math.min(step, height - ty);
            for(int tx=0;tx<width;tx+=step) {
                final int tw = Math.min(step, width - tx);
                // fill input tile
                int k = 0;
                for(int r=0;r<tile;r++) {
                    final int rowOff = r*width;
                    for(int c=0;c<tile;c++) {
                        int sx = reflect(tx - margin + c, width);
                        int p = strip[rowOff + sx];
                        in[k++] = ((p >> 16) & 0xff) * inv255;
                        in[k++] = ((p >> 8) & 0xff) * inv255;
                        in[k++] = (p & 0xff) * inv255;
                    }
                }
                model.run(in, out);
                // write central region
                for(int r=0;r<th;r++) {
                    final int base = ((r + margin)*tile + margin)*3;
                    for(int c=0;c<tw;c++) {
                        int idx = base + c*3;
                        float ir = in[idx], ig = in[idx+1], ib = in[idx+2];
                        float or = ir + strength*(out[idx] - ir);
                        float og = ig + strength*(out[idx+1] - ig);
                        float ob = ib + strength*(out[idx+2] - ib);
                        int origAlpha = strip[(r + margin)*width + tx + c] & 0xff000000;
                        outRow[c] = origAlpha | (toByte(or) << 16) | (toByte(og) << 8) | toByte(ob);
                    }
                    dst.setPixels(outRow, 0, step, tx, ty + r, tw, 1);
                }
                done++;
                if( listener != null && !listener.onProgress(done, total) )
                    return false;
            }
        }
        return true;
    }

    /**
     * Runs a super resolution model (output tile = scale * input tile) over a low resolution image
     * and writes the result box-downsampled by {@code down} into dst, so the full size
     * super-resolved image never has to be held in memory. dst must be
     * (lr.width * scale / down) x (lr.height * scale / down).
     *
     * <p>If {@code original} is non-null (same size as dst) the result is blended with it:
     * out = original + strength * (sr - original).</p>
     */
    public static boolean runTiledSR(PixelImage lr, PixelImage dst, TileModel model, int scale, int down, int margin,
                                     PixelImage original, float strength, ProgressListener listener) {
        final int width = lr.getWidth();
        final int height = lr.getHeight();
        final int tile = model.getTileSize();
        final int step = tile - 2*margin;
        if( step <= 0 )
            throw new IllegalArgumentException("margin too large for tile size");
        if( (step*scale) % down != 0 || scale % down != 0 )
            throw new IllegalArgumentException("scale must be divisible by down");
        final int f = scale / down; // dst pixels per lr pixel
        if( dst.getWidth() != width*f || dst.getHeight() != height*f )
            throw new IllegalArgumentException("dst size mismatch");
        if( original != null && (original.getWidth() != dst.getWidth() || original.getHeight() != dst.getHeight()) )
            throw new IllegalArgumentException("original size mismatch");
        strength = Math.max(0.0f, Math.min(1.0f, strength));
        final int outTile = tile*scale;
        final int total = countTiles(width, height, tile, margin);
        final float[] in = new float[tile*tile*3];
        final float[] out = new float[outTile*outTile*3];
        final int dstStep = step*f;
        final int[] outRow = new int[dstStep];
        final int[] origRow = new int[dstStep];
        final int[] strip = new int[tile*width];
        final int[] rowBuf = new int[width];
        final float inv255 = 1.0f/255.0f;
        final float norm = 1.0f/(down*down);
        int done = 0;
        for(int ty=0;ty<height;ty+=step) {
            for(int r=0;r<tile;r++) {
                lr.getPixels(rowBuf, 0, width, 0, reflect(ty - margin + r, height), width, 1);
                System.arraycopy(rowBuf, 0, strip, r*width, width);
            }
            final int th = Math.min(step, height - ty);
            for(int tx=0;tx<width;tx+=step) {
                final int tw = Math.min(step, width - tx);
                int k = 0;
                for(int r=0;r<tile;r++) {
                    final int rowOff = r*width;
                    for(int c=0;c<tile;c++) {
                        int p = strip[rowOff + reflect(tx - margin + c, width)];
                        in[k++] = ((p >> 16) & 0xff) * inv255;
                        in[k++] = ((p >> 8) & 0xff) * inv255;
                        in[k++] = (p & 0xff) * inv255;
                    }
                }
                model.run(in, out);
                final int dh = th*f, dw = tw*f;
                final int dx0 = tx*f, dy0 = ty*f;
                for(int r=0;r<dh;r++) {
                    if( original != null )
                        original.getPixels(origRow, 0, dstStep, dx0, dy0 + r, dw, 1);
                    for(int c=0;c<dw;c++) {
                        float sr = 0, sg = 0, sb = 0;
                        final int oy0 = margin*scale + r*down;
                        final int ox0 = margin*scale + c*down;
                        for(int yy=0;yy<down;yy++) {
                            int base = ((oy0 + yy)*outTile + ox0)*3;
                            for(int xx=0;xx<down;xx++) {
                                sr += out[base]; sg += out[base+1]; sb += out[base+2];
                                base += 3;
                            }
                        }
                        sr *= norm; sg *= norm; sb *= norm;
                        int alpha = 0xff000000;
                        if( original != null ) {
                            int o = origRow[c];
                            alpha = o & 0xff000000;
                            float or = ((o >> 16) & 0xff)*inv255, og = ((o >> 8) & 0xff)*inv255, ob = (o & 0xff)*inv255;
                            sr = or + strength*(sr - or);
                            sg = og + strength*(sg - og);
                            sb = ob + strength*(sb - ob);
                        }
                        outRow[c] = alpha | (toByte(sr) << 16) | (toByte(sg) << 8) | toByte(sb);
                    }
                    dst.setPixels(outRow, 0, dstStep, dx0, dy0 + r, dw, 1);
                }
                done++;
                if( listener != null && !listener.onProgress(done, total) )
                    return false;
            }
        }
        return true;
    }

    /** over = base + strength * (over - base), in place on over. Images must be the same size. */
    public static void blend(PixelImage base, PixelImage over, float strength) {
        if( strength >= 1.0f )
            return;
        int width = over.getWidth(), height = over.getHeight();
        if( base.getWidth() != width || base.getHeight() != height )
            throw new IllegalArgumentException("size mismatch");
        int[] a = new int[width], b = new int[width];
        for(int y=0;y<height;y++) {
            base.getPixels(a, 0, width, 0, y, width, 1);
            over.getPixels(b, 0, width, 0, y, width, 1);
            for(int x=0;x<width;x++) {
                int pa = a[x], pb = b[x];
                int r = clamp255(((pa >> 16) & 0xff) + strength*(((pb >> 16) & 0xff) - ((pa >> 16) & 0xff)));
                int g = clamp255(((pa >> 8) & 0xff) + strength*(((pb >> 8) & 0xff) - ((pa >> 8) & 0xff)));
                int bl = clamp255((pa & 0xff) + strength*((pb & 0xff) - (pa & 0xff)));
                b[x] = (pa & 0xff000000) | (r << 16) | (g << 8) | bl;
            }
            over.setPixels(b, 0, width, 0, y, width, 1);
        }
    }

    static int toByte(float v) {
        int i = (int)(v*255.0f + 0.5f);
        return i < 0 ? 0 : (Math.min(i, 255));
    }

    // ------------------------------------------------------------------------------------------
    // Tone and colour stage
    // ------------------------------------------------------------------------------------------

    /** Scene categories recognised by the scene detector. */
    public enum Scene {
        GENERAL, PORTRAIT, FOOD, LANDSCAPE, NATURE, NIGHT, DOCUMENT
    }

    /** Parameters for the tone / colour stage. */
    public static final class ToneParams {
        /** Amount of auto levels (0 = off, 1 = full stretch between percentiles). */
        public float levels = 0.6f;
        /** Strength of the S-curve contrast boost (0..1). */
        public float contrast = 0.15f;
        /** Vibrance: saturation boost weighted towards less saturated pixels (-1..1). */
        public float vibrance = 0.15f;
        /** Shadow lift (0..1), useful for night shots. */
        public float shadows = 0.0f;
        /** How much of the estimated colour cast to remove (0 = keep, 1 = fully neutral). */
        public float whiteBalance = 0.5f;
        /** Amount to protect skin tones from saturation changes (0..1). */
        public float skinProtect = 0.0f;

        public static ToneParams forScene(Scene scene, float strength) {
            ToneParams p = new ToneParams();
            switch( scene ) {
                case PORTRAIT:
                    p.levels = 0.5f; p.contrast = 0.08f; p.vibrance = 0.08f; p.skinProtect = 1.0f; p.whiteBalance = 0.4f;
                    break;
                case FOOD:
                    p.levels = 0.6f; p.contrast = 0.15f; p.vibrance = 0.28f; p.skinProtect = 0.0f; p.whiteBalance = 0.25f; // keep appetising warmth
                    break;
                case LANDSCAPE:
                    p.levels = 0.7f; p.contrast = 0.2f; p.vibrance = 0.25f; p.whiteBalance = 0.25f; // don't kill sunsets
                    break;
                case NATURE:
                    p.levels = 0.6f; p.contrast = 0.15f; p.vibrance = 0.22f; p.whiteBalance = 0.35f; // dominant greens bias grey world
                    break;
                case NIGHT:
                    p.levels = 0.35f; p.contrast = 0.05f; p.vibrance = 0.05f; p.shadows = 0.25f;
                    break;
                case DOCUMENT:
                    p.levels = 0.9f; p.contrast = 0.3f; p.vibrance = -0.1f; p.whiteBalance = 0.8f; // paper should be white
                    break;
                case GENERAL:
                default:
                    break;
            }
            strength = Math.max(0.0f, Math.min(1.5f, strength));
            p.levels *= strength;
            p.contrast *= strength;
            p.vibrance *= strength;
            p.shadows *= strength;
            p.whiteBalance = Math.min(1.0f, p.whiteBalance * strength);
            return p;
        }
    }

    /** Lookup table built from image statistics, applied per pixel. */
    public static final class ToneMapper {
        final int[] lut = new int[256];
        final int[] lutR = new int[256], lutG = new int[256], lutB = new int[256];
        final float[] gains;
        final ToneParams params;

        ToneMapper(ToneParams params, int[] lut, float[] gains) {
            this.params = params;
            this.gains = gains.clone();
            System.arraycopy(lut, 0, this.lut, 0, 256);
            for(int i=0;i<256;i++) {
                lutR[i] = lut[Math.min(255, (int)(i*gains[0] + 0.5f))];
                lutG[i] = lut[Math.min(255, (int)(i*gains[1] + 0.5f))];
                lutB[i] = lut[Math.min(255, (int)(i*gains[2] + 0.5f))];
            }
        }

        public int[] getLut() {
            return lut.clone();
        }

        public float[] getGains() {
            return gains.clone();
        }

        /** Applies white balance, the tone curve and vibrance to ARGB pixels in place. */
        public void apply(int[] pixels, int offset, int count) {
            final float vib = params.vibrance;
            final float skin = params.skinProtect;
            for(int i=offset;i<offset+count;i++) {
                int p = pixels[i];
                int r = lutR[(p >> 16) & 0xff];
                int g = lutG[(p >> 8) & 0xff];
                int b = lutB[p & 0xff];
                if( vib != 0.0f ) {
                    int max = Math.max(r, Math.max(g, b));
                    int min = Math.min(r, Math.min(g, b));
                    if( max > 0 ) {
                        float sat = (max - min) / (float)max;
                        float amount = vib * (1.0f - sat);
                        if( skin > 0.0f && isSkin(r, g, b) )
                            amount *= (1.0f - skin);
                        if( amount != 0.0f ) {
                            float luma = 0.299f*r + 0.587f*g + 0.114f*b;
                            r = clamp255(luma + (r - luma)*(1.0f + amount));
                            g = clamp255(luma + (g - luma)*(1.0f + amount));
                            b = clamp255(luma + (b - luma)*(1.0f + amount));
                        }
                    }
                }
                pixels[i] = (p & 0xff000000) | (r << 16) | (g << 8) | b;
            }
        }
    }

    /**
     * Estimates per channel gains that remove part of the colour cast (e.g. the yellow/orange of
     * incandescent lamps), using a grey-world estimate over mid-tone, non clipped pixels.
     * Gains are limited and normalised so that luminance is preserved.
     * @param amount 0 = no correction, 1 = full grey world
     */
    public static float[] estimateWhiteBalanceGains(PixelImage image, int sampleStep, float amount) {
        float[] none = {1.0f, 1.0f, 1.0f};
        if( amount <= 0.0f )
            return none;
        int width = image.getWidth(), height = image.getHeight();
        int[] row = new int[width];
        sampleStep = Math.max(1, sampleStep);
        double sr = 0, sg = 0, sb = 0;
        long n = 0;
        for(int y=0;y<height;y+=sampleStep) {
            image.getPixels(row, 0, width, 0, y, width, 1);
            for(int x=0;x<width;x+=sampleStep) {
                int p = row[x];
                int r = (p >> 16) & 0xff, g = (p >> 8) & 0xff, b = p & 0xff;
                int l = (299*r + 587*g + 114*b) / 1000;
                int max = Math.max(r, Math.max(g, b));
                if( l > 38 && l < 235 && max < 250 ) {
                    sr += r; sg += g; sb += b;
                    n++;
                }
            }
        }
        if( n < 16 )
            return none;
        double mr = sr/n, mg = sg/n, mb = sb/n;
        if( mr < 1 || mg < 1 || mb < 1 )
            return none;
        double mean = (mr + mg + mb) / 3.0;
        float[] g = new float[3];
        double[] m = {mr, mg, mb};
        for(int c=0;c<3;c++) {
            double full = mean / m[c];
            double v = 1.0 + amount * (full - 1.0);
            g[c] = (float)Math.max(0.8, Math.min(1.25, v));
        }
        // preserve luminance
        float lum = 0.299f*g[0] + 0.587f*g[1] + 0.114f*g[2];
        for(int c=0;c<3;c++)
            g[c] /= lum;
        return g;
    }

    static boolean isSkin(int r, int g, int b) {
        // classic RGB skin heuristic (Kovac et al.), works well enough for protecting colours
        return r > 95 && g > 40 && b > 20 && r > g && r > b && (r - Math.min(g, b)) > 15 && Math.abs(r - g) > 15;
    }

    static int clamp255(float v) {
        int i = (int)(v + 0.5f);
        return i < 0 ? 0 : Math.min(i, 255);
    }

    /**
     * Builds the tone mapper from a luminance histogram (256 bins) of the image.
     */
    public static ToneMapper buildToneMapper(long[] histogram, ToneParams params) {
        return buildToneMapper(histogram, params, new float[]{1.0f, 1.0f, 1.0f});
    }

    /** As above, additionally applying the white balance gains (see {@link #estimateWhiteBalanceGains}). */
    public static ToneMapper buildToneMapper(long[] histogram, ToneParams params, float[] wbGains) {
        long total = 0;
        for(long h : histogram)
            total += h;
        int lo = 0, hi = 255;
        if( total > 0 ) {
            long loCount = (long)(total * 0.003);
            long hiCount = (long)(total * 0.997);
            long acc = 0;
            lo = -1;
            hi = 255;
            for(int i=0;i<256;i++) {
                acc += histogram[i];
                if( lo < 0 && acc > loCount )
                    lo = i;
                if( acc >= hiCount ) {
                    hi = i;
                    break;
                }
            }
            if( lo < 0 )
                lo = 0;
        }
        // limit the stretch so we never do anything drastic (e.g. on a deliberately dark/low key shot)
        lo = Math.min(lo, 40);
        hi = Math.max(hi, 200);
        if( hi <= lo ) {
            lo = 0;
            hi = 255;
        }
        float black = params.levels * lo;
        float white = 255.0f - params.levels * (255 - hi);
        int[] lut = new int[256];
        for(int i=0;i<256;i++) {
            float x = (i - black) / (white - black);
            x = Math.max(0.0f, Math.min(1.0f, x));
            if( params.shadows > 0.0f ) {
                // lift shadows: blend with a gamma curve
                float lifted = (float)Math.pow(x, 1.0/(1.0 + 1.5*params.shadows));
                x = x + (lifted - x) * (1.0f - x);
            }
            if( params.contrast != 0.0f ) {
                // smoothstep based S-curve, mixed with identity
                float s = x*x*(3.0f - 2.0f*x);
                x = x + params.contrast*(s - x);
            }
            lut[i] = clamp255(x*255.0f);
        }
        // guarantee monotonic
        for(int i=1;i<256;i++) {
            if( lut[i] < lut[i-1] )
                lut[i] = lut[i-1];
        }
        return new ToneMapper(params, lut, wbGains);
    }

    /** Computes a luminance histogram, sampling every {@code sampleStep}-th pixel in each direction. */
    public static long[] luminanceHistogram(PixelImage image, int sampleStep) {
        long[] hist = new long[256];
        int width = image.getWidth();
        int height = image.getHeight();
        int[] row = new int[width];
        sampleStep = Math.max(1, sampleStep);
        for(int y=0;y<height;y+=sampleStep) {
            image.getPixels(row, 0, width, 0, y, width, 1);
            for(int x=0;x<width;x+=sampleStep) {
                int p = row[x];
                int l = (299*((p >> 16) & 0xff) + 587*((p >> 8) & 0xff) + 114*(p & 0xff)) / 1000;
                hist[l]++;
            }
        }
        return hist;
    }

    /** Applies the tone mapper to the whole image, row by row. */
    public static void applyTone(PixelImage image, ToneMapper mapper) {
        int width = image.getWidth();
        int height = image.getHeight();
        int rows = Math.max(1, Math.min(height, (1 << 20) / Math.max(1, width)));
        int[] buf = new int[width*rows];
        for(int y=0;y<height;y+=rows) {
            int h = Math.min(rows, height - y);
            image.getPixels(buf, 0, width, 0, y, width, h);
            mapper.apply(buf, 0, width*h);
            image.setPixels(buf, 0, width, 0, y, width, h);
        }
    }

    /** Mean luminance (0..255) from a histogram. */
    public static float meanLuminance(long[] hist) {
        long total = 0, sum = 0;
        for(int i=0;i<256;i++) {
            total += hist[i];
            sum += hist[i] * i;
        }
        return total == 0 ? 0.0f : sum / (float)total;
    }

    /**
     * Maps image labels (e.g. from ML Kit image labelling) plus capture metadata to a scene.
     * @param labels lower case labels, in decreasing confidence order
     * @param confidences confidence for each label
     * @param iso capture ISO, or -1 if unknown
     * @param meanLuma mean luminance of the image (0..255)
     * @param faces number of faces detected, or -1 if unknown
     */
    public static Scene classifyScene(String[] labels, float[] confidences, int iso, float meanLuma, int faces) {
        if( (iso >= 1600) || (meanLuma < 50.0f && iso >= 400) || meanLuma < 35.0f )
            return Scene.NIGHT;
        if( faces > 0 )
            return Scene.PORTRAIT;
        float portrait = 0, food = 0, landscape = 0, nature = 0, document = 0;
        if( labels != null ) {
            for(int i=0;i<labels.length;i++) {
                String l = labels[i] == null ? "" : labels[i].toLowerCase(java.util.Locale.US);
                float c = confidences != null && i < confidences.length ? confidences[i] : 0.5f;
                if( l.equals("selfie") || l.equals("smile") || l.equals("beard") || l.equals("eyelash") || l.equals("person") || l.equals("hair") || l.equals("skin") )
                    portrait += c;
                else if( l.equals("food") || l.equals("cuisine") || l.equals("dish") || l.equals("cake") || l.equals("fruit") || l.equals("vegetable") || l.equals("bread") || l.equals("pizza") || l.equals("sushi") || l.equals("coffee") || l.equals("cookie") || l.equals("dessert") )
                    food += c;
                else if( l.equals("sky") || l.equals("mountain") || l.equals("lake") || l.equals("sea") || l.equals("beach") || l.equals("cloud") || l.equals("sunset") || l.equals("river") || l.equals("skyline") || l.equals("waterfall") || l.equals("desert") || l.equals("bridge") )
                    landscape += c;
                else if( l.equals("flower") || l.equals("plant") || l.equals("tree") || l.equals("garden") || l.equals("petal") || l.equals("leaf") || l.equals("grass") || l.equals("forest") || l.equals("animal") || l.equals("bird") || l.equals("pet") || l.equals("dog") || l.equals("cat") )
                    nature += c;
                else if( l.equals("paper") || l.equals("poster") || l.equals("menu") || l.equals("receipt") || l.equals("newspaper") || l.equals("text") || l.equals("handwriting") || l.equals("whiteboard") || l.equals("document") )
                    document += c;
            }
        }
        float best = 0.45f; // minimum confidence to leave GENERAL
        Scene scene = Scene.GENERAL;
        if( portrait > best ) { best = portrait; scene = Scene.PORTRAIT; }
        if( food > best ) { best = food; scene = Scene.FOOD; }
        if( document > best ) { best = document; scene = Scene.DOCUMENT; }
        if( landscape > best ) { best = landscape; scene = Scene.LANDSCAPE; }
        if( nature > best ) { scene = Scene.NATURE; }
        return scene;
    }
}
