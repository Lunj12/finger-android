package com.example.lunj.fingerrecognition;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.TextView;

public class RecognizationActivity extends AppCompatActivity {

    // class instance variables
    int recognization_result = -1;
    int historical_sum = 0;
    char RADIO_OPTIONS = 'N'; // 'N' : None, 'D' : add digit, 'S' : sum up

    // class instances view
    RadioGroup rg;
    TextView result_field, sum_field;
    Button take_another, finish_go;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recognization);

        rg = (RadioGroup) findViewById(R.id.radioGroup);
        result_field = (TextView) findViewById(R.id.result_field);
        sum_field = (TextView) findViewById(R.id.sum_field);
        take_another = (Button) findViewById(R.id.take_another);
        take_another = (Button) findViewById(R.id.take_another);

        // listener for radio group
        rg.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                switch (checkedId) {
                    case R.id.add_digit:
                        // add digit mark
//                        RADIO_OPTIONS = 1;
//                        .setVisibility(View.VISIBLE);
                        break;
                    case R.id.sum_up:
                        // add digit mark
//                        retake.setVisibility(View.VISIBLE);
                        break;
                    case R.id.finish:
                        // finish button appears
//                        finish_go.setVisibility(View.VISIBLE);
                        break;
                }

            }
        });
    }

    public void reLaunchCamera(View v) {

    }

    public void recognize(View view) {
    }

}
