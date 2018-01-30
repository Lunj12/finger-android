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
import android.widget.RadioGroup;
import android.widget.TextView;


public class FirstActivity extends AppCompatActivity {
    public static final int REQUEST_CAPTURE = 1;

    // class instance variables
    int current_result = -1;
    int total_sum = 0;
    int RADIO_OPTIONS = 0; // 0 : None, 1 : add digit, 2 : sum up
    int[] pixels = new int[64 * 64];
    float[] flattenedImage = new float[64 * 64 * 3];

    // class instances view
    ImageView result_photo;
    Button take, retake, next_1, next_2, finish_go;
    RadioGroup rg;
    EditText welcome;
    TextView result_field, sum_field;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_first);

        // find views and set initial visibility
        take = (Button) findViewById(R.id.take);
        retake = (Button) findViewById(R.id.retake);
        next_1 = (Button) findViewById(R.id.next_1);
        next_2 = (Button) findViewById(R.id.next_2);
        finish_go = (Button) findViewById(R.id.finish_go);
        rg = (RadioGroup) findViewById(R.id.radioGroup);
        result_photo = (ImageView) findViewById(R.id.imageView);
        welcome = (EditText) findViewById(R.id.welcome);
        result_field = (TextView) findViewById(R.id.result_field);
        sum_field = (TextView) findViewById(R.id.sum_field);

        initVisibility();

        // listener for radio group
        rg.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                switch (checkedId) {
                    case R.id.add_digit:
                        // add digit mark
                        RADIO_OPTIONS = 1;
                        retake.setVisibility(View.VISIBLE);
                        break;
                    case R.id.sum_up:
                        // add digit mark
                        retake.setVisibility(View.VISIBLE);
                        break;
                    case R.id.finish:
                        // finish button appears
                        finish_go.setVisibility(View.VISIBLE);
                        break;
                }

            }
        });

        // check camera availability
        if (!hasCamera()) {
            take.setEnabled(false);
        }

    }

    public void initVisibility() {
        // set initial visibilities
        take.setVisibility(View.VISIBLE);
        retake.setVisibility(View.GONE);
        next_1.setVisibility(View.GONE);
        next_2.setVisibility(View.GONE);
        finish_go.setVisibility(View.GONE);
        rg.setVisibility(View.GONE);
        result_photo.setVisibility(View.VISIBLE);
        welcome.setVisibility(View.VISIBLE);
        result_field.setVisibility(View.GONE);
        sum_field.setVisibility(View.GONE);
    }

    public boolean hasCamera() {
        // default front camera, how about a back one?
        return getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY);
    }

    public void launchCamera(View v) {
        Intent i = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(i, REQUEST_CAPTURE);
    }

    public void reLaunchCamera(View v) {

    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_CAPTURE && resultCode == RESULT_OK) {
            //


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


    public void recognize(View view) {
    }
}
