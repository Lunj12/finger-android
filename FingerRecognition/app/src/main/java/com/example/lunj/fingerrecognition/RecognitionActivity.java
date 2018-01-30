package com.example.lunj.fingerrecognition;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import com.example.lunj.fingerrecognition.models.Classification;
import com.example.lunj.fingerrecognition.models.Classifier;
import com.example.lunj.fingerrecognition.models.TensorFlowClassifier;


public class RecognitionActivity extends AppCompatActivity {

    // class instance variables
    float[] imageArray;
    int recognition_result = -1;
    int historical_sum;
    char RADIO_OPTIONS; // 'N' : None, 'A', abandoned, S' : sum-up

    // class instances view
    RadioGroup rg;
    TextView result_field, sum_field;
    Button confirm, take_another, finish_go;
    EditText recog_text, sum_text;

    // tensorflow classifier
    Classifier myClassifier;
    final int IMAGE_ARRAY_LENGTH = 64 * 64 * 3;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recognition);

        // restore saved instance state
        if (savedInstanceState != null) {
            historical_sum = savedInstanceState.getInt("historical_sum");
            RADIO_OPTIONS = savedInstanceState.getChar("RADIO_OPTIONS");
        } else {
            historical_sum = 0;
            RADIO_OPTIONS = 'N';
        }

        // get views and setup initial properties
        rg = (RadioGroup) findViewById(R.id.radioGroup);
        result_field = (TextView) findViewById(R.id.result_field);
        sum_field = (TextView) findViewById(R.id.sum_field);
        confirm = (Button) findViewById(R.id.confirm);
        take_another = (Button) findViewById(R.id.take_another);
        finish_go = (Button) findViewById(R.id.finish_go);
        recog_text = (EditText) findViewById(R.id.recog_result);
        sum_text = (EditText) findViewById(R.id.histo_sum);

        recog_text.setKeyListener(null);
        sum_text.setKeyListener(null);
        take_another.setAlpha(.5f);
        take_another.setClickable(false);
        finish_go.setAlpha(.5f);
        finish_go.setClickable(false);
        sum_field.setText("" + historical_sum);

        // retrieve image array from intent
        Intent caller = getIntent();
        imageArray = caller.getFloatArrayExtra("imageArray");

        // recognize and calculate;
        loadModel();
        TFRecognize();
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


    public void confirmOption(View view) {
        int checkedId = rg.getCheckedRadioButtonId();
        // Options for radio group
        switch (checkedId) {
            case -1:
                Toast.makeText(getApplicationContext(), "No option selected!", Toast.LENGTH_SHORT).show();
                return;
            case R.id.keep_result:
                // toast
                Toast.makeText(getApplicationContext(), "Result Kept! Go to finish!", Toast.LENGTH_SHORT).show();
                // manipulate numbers
                historical_sum += recognition_result;
                // set button
                finish_go.setAlpha(1f);
                finish_go.setClickable(true);
                break;
            case R.id.abandon_result:
                // add abandon mark 'A'
                RADIO_OPTIONS = 'A';
                // toast
                Toast.makeText(getApplicationContext(), "Result Abandoned!", Toast.LENGTH_SHORT).show();
                // manipulate numbers
                recognition_result = -1;
                // set button
                finish_go.setAlpha(1f);
                finish_go.setClickable(true);
                take_another.setAlpha(1f);
                take_another.setClickable(true);
                break;
            case R.id.sum_up:
                // sum up mark 'S'
                RADIO_OPTIONS = 'S';
                Toast.makeText(getApplicationContext(), "Result Kept! Go to another", Toast.LENGTH_SHORT).show();
                // manipulate numbers
                historical_sum += recognition_result;
                // set button
                take_another.setAlpha(1f);
                take_another.setClickable(true);
                break;
        }

        // reset the two text fields
        resetTextFields("" + recognition_result, "" + historical_sum);
        // disable the confirm button
        confirm.setAlpha(.5f);
        confirm.setClickable(false);
        // disable all children in the radio group
        for (int i = 0; i < rg.getChildCount(); i++) {
            rg.getChildAt(i).setEnabled(false);
        }
    }

    public void resetTextFields(String result_str, String sum_str) {
        if (result_str != null) {
            result_field.setText(result_str);
        }

        if (sum_str != null) {
            sum_field.setText(sum_str);
        }
    }

    private void loadModel() {
        try {
            myClassifier = TensorFlowClassifier.create(getAssets(), "TensorFlow",
                    "opt_finger_linear.pb", "labels.txt", IMAGE_ARRAY_LENGTH,
                    "X", "predicts", false);

        } catch (final Exception e) {
            throw new RuntimeException("Error initializing classifier!", e);
        }

    }

    public void TFRecognize() {
        String text = "";
        final Classification res = myClassifier.recognize(imageArray);
        if (res.getLabel() == null) {
            text += myClassifier.name() + ": ?/n";
        } else {
            recognition_result = Integer.parseInt(res.getLabel());
            text += String.format("%s: %s, %f", myClassifier.name(), res.getLabel(), res.getConf());
        }

        resetTextFields(text, null);
    }
}

