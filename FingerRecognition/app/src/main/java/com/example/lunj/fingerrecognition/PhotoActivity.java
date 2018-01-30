package com.example.lunj.fingerrecognition;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;


public class PhotoActivity extends AppCompatActivity {
    public static final int REQUEST_CAPTURE = 1;

    // class instance variables
    int[] pixels = new int[64 * 64];
    float[] flattenedImage = new float[64 * 64 * 3];

    // class instances view
    ImageView result_photo;
    Button take, next_1;
    EditText welcome;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_photo);

        // find views and set initial visibility
        take = (Button) findViewById(R.id.take);
        next_1 = (Button) findViewById(R.id.next_1);
        result_photo = (ImageView) findViewById(R.id.imageView);
        welcome = (EditText) findViewById(R.id.welcome);

        // check camera availability
        if (!hasCamera()) {
            take.setEnabled(false);
        }

    }


    public boolean hasCamera() {
        // default front camera, how about a back one?
        return getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY);
    }

    public void launchCamera(View v) {
        Intent i = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(i, REQUEST_CAPTURE);
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            Bitmap photo = (Bitmap) extras.get("data");
            Bitmap sizedPhoto = Bitmap.createScaledBitmap(photo, 64, 64, true);
            int width = sizedPhoto.getWidth();
            int height = sizedPhoto.getWidth();
            sizedPhoto.getPixels(pixels, 0, width, 0, 0, width, height);
            flattenImage();
            result_photo.setImageBitmap(photo);
        }

    }

    protected void flattenImage() {
        for (int i = 0; i < pixels.length; i++) {
            int redValue = Color.red(pixels[i]);
            int blueValue = Color.blue(pixels[i]);
            int greenValue = Color.green(pixels[i]);
            this.flattenedImage[3 * i] = (float) redValue / 255;
            this.flattenedImage[3 * i + 1] = (float) blueValue / 255;
            this.flattenedImage[3 * i + 2] = (float) greenValue / 255;
        }
    }

}
