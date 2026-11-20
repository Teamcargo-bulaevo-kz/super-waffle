package net.sourceforge.opencamera.ai;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Parcelable;
import android.provider.MediaStore;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.exifinterface.media.ExifInterface;

import net.sourceforge.opencamera.R;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * "Enhance photo" screen: lets the user run the NeuroCam AI pipeline (including generative
 * AI detail) on any existing photo, e.g. shared from the gallery. The result is saved as a new
 * photo in Pictures/NeuroCam; the original is never modified.
 */
public class EnhanceActivity extends Activity {
    private static final String TAG = "EnhanceActivity";
    private static final int REQUEST_PICK = 1;
    /** Max pixels processed (larger photos are downscaled to avoid running out of memory). */
    private static final int MAX_PIXELS = 16_000_000;

    /** Intent extra (boolean): start processing immediately, used by tests. */
    public static final String EXTRA_AUTO_START = "neurocam_auto_start";
    /** Intent extra (boolean): use AI detail, overrides the checkbox default. */
    public static final String EXTRA_DETAIL = "neurocam_detail";

    /** For tests. */
    public static volatile Uri lastSavedUri;
    public static volatile NeuroEnhancer.Result lastResult;

    private Uri sourceUri;
    private Bitmap sourceBitmap;
    private Bitmap resultBitmap;
    private Uri savedUri;
    private volatile boolean processing;

    private ImageView imageView;
    private TextView statusView;
    private ProgressBar progressBar;
    private CheckBox detailCheck;
    private CheckBox wbCheck;
    private Button enhanceButton;
    private Button shareButton;
    private Button compareButton;
    private boolean showingOriginal;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();

        Intent intent = getIntent();
        sourceUri = extractUri(intent);
        detailCheck.setChecked(intent.getBooleanExtra(EXTRA_DETAIL, true));
        if( sourceUri == null ) {
            Intent pick = new Intent(Intent.ACTION_GET_CONTENT);
            pick.setType("image/*");
            pick.addCategory(Intent.CATEGORY_OPENABLE);
            try {
                startActivityForResult(Intent.createChooser(pick, getString(R.string.ai_enhance_pick)), REQUEST_PICK);
            }
            catch(Exception e) {
                Log.e(TAG, "no picker", e);
                statusView.setText(R.string.ai_enhance_failed_load);
            }
        }
        else {
            loadSource(intent.getBooleanExtra(EXTRA_AUTO_START, false));
        }
    }

    private static Uri extractUri(Intent intent) {
        if( intent == null )
            return null;
        if( Intent.ACTION_SEND.equals(intent.getAction()) ) {
            Parcelable p = intent.getParcelableExtra(Intent.EXTRA_STREAM);
            if( p instanceof Uri )
                return (Uri)p;
        }
        return intent.getData();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if( requestCode == REQUEST_PICK ) {
            if( resultCode == RESULT_OK && data != null && data.getData() != null ) {
                sourceUri = data.getData();
                loadSource(false);
            }
            else if( sourceBitmap == null ) {
                finish();
            }
        }
    }

    private int dp(int v) {
        return (int)TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics());
    }

    private void buildUi() {
        setTitle(R.string.ai_enhance_title);
        ScrollView scroll = new ScrollView(this);
        scroll.setBackgroundColor(Color.BLACK);
        scroll.setFitsSystemWindows(true);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(16), dp(16), dp(16));
        scroll.addView(root);

        TextView title = new TextView(this);
        title.setText(R.string.ai_enhance_title);
        title.setTextColor(Color.WHITE);
        title.setTextSize(TypedValue.COMPLEX_UNIT_SP, 22);
        root.addView(title);

        imageView = new ImageView(this);
        imageView.setAdjustViewBounds(true);
        imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
        LinearLayout.LayoutParams ilp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(380));
        ilp.topMargin = dp(12);
        root.addView(imageView, ilp);

        statusView = new TextView(this);
        statusView.setTextColor(0xffdddddd);
        statusView.setGravity(Gravity.CENTER_HORIZONTAL);
        statusView.setPadding(0, dp(8), 0, dp(8));
        root.addView(statusView);

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(1000);
        progressBar.setVisibility(View.GONE);
        root.addView(progressBar, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        detailCheck = new CheckBox(this);
        detailCheck.setText(R.string.ai_enhance_detail);
        detailCheck.setTextColor(Color.WHITE);
        root.addView(detailCheck);

        wbCheck = new CheckBox(this);
        wbCheck.setText(R.string.preference_ai_white_balance);
        wbCheck.setTextColor(Color.WHITE);
        wbCheck.setChecked(true);
        root.addView(wbCheck);

        enhanceButton = new Button(this);
        enhanceButton.setText(R.string.ai_enhance_start);
        enhanceButton.setEnabled(false);
        enhanceButton.setOnClickListener(v -> startProcessing());
        root.addView(enhanceButton);

        compareButton = new Button(this);
        compareButton.setText(R.string.ai_enhance_compare);
        compareButton.setVisibility(View.GONE);
        compareButton.setOnClickListener(v -> {
            showingOriginal = !showingOriginal;
            imageView.setImageBitmap(showingOriginal ? sourceBitmap : resultBitmap);
            compareButton.setText(showingOriginal ? R.string.ai_enhance_show_result : R.string.ai_enhance_compare);
        });
        root.addView(compareButton);

        shareButton = new Button(this);
        shareButton.setText(R.string.share);
        shareButton.setVisibility(View.GONE);
        shareButton.setOnClickListener(v -> {
            if( savedUri != null ) {
                Intent share = new Intent(Intent.ACTION_SEND);
                share.setType("image/jpeg");
                share.putExtra(Intent.EXTRA_STREAM, savedUri);
                share.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                startActivity(Intent.createChooser(share, getString(R.string.share)));
            }
        });
        root.addView(shareButton);

        TextView note = new TextView(this);
        note.setText(R.string.ai_enhance_note);
        note.setTextColor(0xff999999);
        note.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        note.setPadding(0, dp(12), 0, 0);
        root.addView(note);

        setContentView(scroll);
    }

    private void loadSource(final boolean autoStart) {
        statusView.setText(R.string.ai_enhance_loading);
        new Thread(() -> {
            Bitmap bmp = null;
            try {
                bmp = decode(getContentResolver(), sourceUri);
            }
            catch(Throwable e) {
                Log.e(TAG, "failed to load " + sourceUri, e);
            }
            final Bitmap loaded = bmp;
            runOnUiThread(() -> {
                if( isFinishing() )
                    return;
                if( loaded == null ) {
                    statusView.setText(R.string.ai_enhance_failed_load);
                    return;
                }
                sourceBitmap = loaded;
                imageView.setImageBitmap(loaded);
                statusView.setText(getString(R.string.ai_enhance_ready, loaded.getWidth(), loaded.getHeight()));
                enhanceButton.setEnabled(true);
                if( autoStart )
                    startProcessing();
            });
        }, "EnhanceLoad").start();
    }

    /** Decodes the image, applying EXIF orientation and limiting the size. */
    static Bitmap decode(ContentResolver resolver, Uri uri) throws IOException {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try( InputStream in = resolver.openInputStream(uri) ) {
            BitmapFactory.decodeStream(in, null, bounds);
        }
        if( bounds.outWidth <= 0 || bounds.outHeight <= 0 )
            return null;
        int sample = 1;
        while( (long)(bounds.outWidth/sample) * (bounds.outHeight/sample) > MAX_PIXELS )
            sample *= 2;
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = sample;
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        options.inMutable = true;
        Bitmap bmp;
        try( InputStream in = resolver.openInputStream(uri) ) {
            bmp = BitmapFactory.decodeStream(in, null, options);
        }
        if( bmp == null )
            return null;
        int orientation = ExifInterface.ORIENTATION_NORMAL;
        try( InputStream in = resolver.openInputStream(uri) ) {
            if( in != null )
                orientation = new ExifInterface(in).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
        }
        catch(Exception e) {
            Log.e(TAG, "failed to read exif", e);
        }
        Matrix m = new Matrix();
        switch( orientation ) {
            case ExifInterface.ORIENTATION_ROTATE_90: m.postRotate(90); break;
            case ExifInterface.ORIENTATION_ROTATE_180: m.postRotate(180); break;
            case ExifInterface.ORIENTATION_ROTATE_270: m.postRotate(270); break;
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL: m.postScale(-1, 1); break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL: m.postScale(1, -1); break;
            default: break;
        }
        if( !m.isIdentity() ) {
            Bitmap rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), m, true);
            if( rotated != bmp ) {
                bmp.recycle();
                bmp = rotated;
            }
        }
        if( bmp.getConfig() != Bitmap.Config.ARGB_8888 ) {
            Bitmap c = bmp.copy(Bitmap.Config.ARGB_8888, true);
            bmp.recycle();
            bmp = c;
        }
        return bmp;
    }

    private void startProcessing() {
        if( processing || sourceBitmap == null )
            return;
        processing = true;
        enhanceButton.setEnabled(false);
        detailCheck.setEnabled(false);
        wbCheck.setEnabled(false);
        progressBar.setProgress(0);
        progressBar.setVisibility(View.VISIBLE);
        statusView.setText(R.string.ai_enhance_processing);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        final NeuroEnhancer.Options options = NeuroEnhancer.Options.fromPreferences(this);
        options.detail = detailCheck.isChecked();
        options.whiteBalance = wbCheck.isChecked();
        final long start = System.currentTimeMillis();
        options.progress = fraction -> runOnUiThread(() -> {
            progressBar.setProgress((int)(fraction*1000));
            long elapsed = (System.currentTimeMillis() - start) / 1000;
            statusView.setText(getString(R.string.ai_enhance_progress, (int)(fraction*100), elapsed));
        });
        final Bitmap src = sourceBitmap;
        new Thread(() -> {
            NeuroEnhancer.Result result = NeuroEnhancer.getInstance(this).enhance(src, -1, options);
            Uri saved = null;
            if( result.bitmap != null && result.bitmap != src ) {
                try {
                    saved = save(result.bitmap);
                }
                catch(Throwable e) {
                    Log.e(TAG, "failed to save", e);
                }
            }
            final Uri savedFinal = saved;
            lastResult = result;
            lastSavedUri = saved;
            runOnUiThread(() -> onProcessed(result, savedFinal));
        }, "EnhanceProcess").start();
    }

    private void onProcessed(NeuroEnhancer.Result result, Uri saved) {
        processing = false;
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        progressBar.setVisibility(View.GONE);
        if( isFinishing() )
            return;
        if( saved == null || result.bitmap == null ) {
            statusView.setText(R.string.ai_enhance_failed);
            enhanceButton.setEnabled(true);
            detailCheck.setEnabled(true);
            wbCheck.setEnabled(true);
            return;
        }
        resultBitmap = result.bitmap;
        savedUri = saved;
        imageView.setImageBitmap(resultBitmap);
        statusView.setText(getString(R.string.ai_enhance_done, result.timeMs / 1000.0f, result.usedGpu ? "GPU" : "CPU"));
        compareButton.setVisibility(View.VISIBLE);
        shareButton.setVisibility(View.VISIBLE);
    }

    private Uri save(Bitmap bitmap) throws IOException {
        String name = "NeuroCam_AI_" + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date()) + ".jpg";
        ContentResolver resolver = getContentResolver();
        if( Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q ) {
            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.DISPLAY_NAME, name);
            values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
            values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/NeuroCam");
            values.put(MediaStore.Images.Media.IS_PENDING, 1);
            Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            if( uri == null )
                throw new IOException("MediaStore insert failed");
            try( OutputStream out = resolver.openOutputStream(uri) ) {
                if( out == null || !bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out) )
                    throw new IOException("compress failed");
            }
            values.clear();
            values.put(MediaStore.Images.Media.IS_PENDING, 0);
            resolver.update(uri, values, null, null);
            return uri;
        }
        else {
            File dir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
            try {
                File pub = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "NeuroCam");
                if( pub.exists() || pub.mkdirs() )
                    dir = pub;
            }
            catch(Exception e) {
                Log.e(TAG, "no public pictures dir", e);
            }
            File file = new File(dir, name);
            try( FileOutputStream out = new FileOutputStream(file) ) {
                if( !bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out) )
                    throw new IOException("compress failed");
            }
            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.DATA, file.getAbsolutePath());
            values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
            Uri uri = null;
            try {
                uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            }
            catch(Exception e) {
                Log.e(TAG, "failed to add to media store", e);
            }
            return uri != null ? uri : Uri.fromFile(file);
        }
    }
}
