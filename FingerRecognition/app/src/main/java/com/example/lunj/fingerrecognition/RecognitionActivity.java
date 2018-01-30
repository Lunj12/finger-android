package com.example.lunj.fingerrecognition;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

public class RecognitionActivity extends AppCompatActivity {

    // class instance variables
    float[] imageArray;
    int recognition_result = -1;
    int historical_sum;
    char RADIO_OPTIONS; // 'N' : None, 'A' : abandon, 'D' : add digit, 'S' : sum up

    // class instances view
    RadioGroup rg;
    TextView result_field, sum_field;
    Button take_another, finish_go;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recognition);

        // restore saved instance state
        if (savedInstanceState!=null) {
            historical_sum = savedInstanceState.getInt("historical_sum");
            RADIO_OPTIONS = savedInstanceState.getChar("RADIO_OPTIONS");
        } else {
            historical_sum = 0;
            RADIO_OPTIONS = 'N';
        }

        // get views
        rg = (RadioGroup) findViewById(R.id.radioGroup);
        result_field = (TextView) findViewById(R.id.result_field);
        sum_field = (TextView) findViewById(R.id.sum_field);
        take_another = (Button) findViewById(R.id.take_another);

        // retrieve image array from intent
        Intent caller = getIntent();
        imageArray = caller.getFloatArrayExtra("imageArray");

        // recognize and calculate;



        // listener for radio group
        rg.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                switch (checkedId) {
                    case R.id.abandon_result:
                        // add abandon mark 'A', take_another button appears
                        RADIO_OPTIONS = 'A';
                        take_another.setVisibility(View.VISIBLE);
                        break;
                    case R.id.add_digit:
                        // add digit mark 'D', take_another button appears
                        RADIO_OPTIONS = 'D';
                        take_another.setVisibility(View.VISIBLE);
                        break;
                    case R.id.sum_up:
                        // sum up mark 'S', take_another button appears
                        RADIO_OPTIONS = 'S';
                        take_another.setVisibility(View.VISIBLE);
                        break;
                    case R.id.finish:
                        // finish_go button appears
                        RADIO_OPTIONS = 'N';
                        finish_go.setVisibility(View.VISIBLE);
                        break;
                }

            }
        });
    }

    @Override
    protected void onSaveInstanceState(Bundle savedInstanceState) {
        super.onSaveInstanceState(savedInstanceState);

        savedInstanceState.putInt("historical_sum", historical_sum);
        savedInstanceState.putChar("RADIO_OPTIONS", RADIO_OPTIONS);
    }

    public void reLaunchCamera(View v) {
        // show a toast message indicating the radio options
        String msg = "";
        switch (RADIO_OPTIONS) {
            case 'A':
                historical_sum -= recognition_result; // abandon old result;
                msg = "Old result abandoned! Go for a new one!";
                break;
            case 'D':
                msg = "Go to add a new digit!";
                break;
            case 'S':
                msg = "Go to add a new value for sum-up!";
                break;
        }
        Toast.makeText(getApplicationContext(), msg, Toast.LENGTH_SHORT).show();

        // start photo activity
        Intent goToPhoto = new Intent();
        goToPhoto.setClass(this, PhotoActivity.class);
        startActivity(goToPhoto);
        finish();
    }



}
