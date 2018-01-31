package com.example.lunj.fingerrecognition;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.NumberPicker;
import android.widget.Toast;

public class SearchActivity extends AppCompatActivity {
    // class instance variables
    int numberToSearch;

    // class instance views
    WebView wv;
    NumberPicker picker;
    Button search, exit;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_search);

        // retrieve numebr to search from intent
        Intent caller = getIntent();
        numberToSearch = caller.getIntExtra("historical_sum", 0);

        // set views
        wv = (WebView) findViewById(R.id.webView);
        picker = (NumberPicker) findViewById(R.id.picker);
        search = (Button) findViewById(R.id.search);
        exit = (Button) findViewById(R.id.exit);
        picker.setMinValue(0);
        picker.setMaxValue(2);
        picker.setDisplayedValues(new String[]{"English", "French", "Chinese"});
        picker.setDescendantFocusability(NumberPicker.FOCUS_BLOCK_DESCENDANTS);

        // inline display and initial navigation
        wv.setWebViewClient(new WebViewClient());
        loadWeb("en");
    }

    public void navigate(View view) {
        int choice = picker.getValue();
        String lang = "";
        if (choice == 0) {
            lang = "en";
            Toast.makeText(getApplicationContext(), "Switched to English Wiki!", Toast.LENGTH_SHORT).show();
        } else if (choice == 1) {
            lang = "fr";
            Toast.makeText(getApplicationContext(), "Passé au français Wiki", Toast.LENGTH_SHORT).show();
        } else if (choice == 2) {
            lang = "zh";
            Toast.makeText(getApplicationContext(), "切换至中文维基", Toast.LENGTH_SHORT).show();
        }
        loadWeb(lang);
    }


    public void loadWeb(String lang) {
        String url = "https://%s.wikipedia.org/wiki/%s";
        url = String.format(url, lang, numberToSearch + "");
        wv.loadUrl(url);
    }

    public void exitProgram(View view) {
        Toast.makeText(getApplicationContext(), "Thanks for using! Bye!", Toast.LENGTH_SHORT).show();
        finish();
    }
}
