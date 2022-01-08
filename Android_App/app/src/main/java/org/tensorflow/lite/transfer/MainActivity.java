package org.tensorflow.lite.transfer;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        findViewById(R.id.main_collect_data_click_view).setOnClickListener(v -> {
            final Intent intent = new Intent(MainActivity.this, DigitList.class);
            startActivity(intent);
        });

        findViewById(R.id.main_test_click_view).setOnClickListener(v -> {
            final Intent intent = new Intent(MainActivity.this, ModelList.class);
            startActivity(intent);
        });

        findViewById(R.id.main_transfer_learning_click_view).setOnClickListener(v -> {
            final Intent intent = new Intent(MainActivity.this, TrainModelList.class);
            startActivity(intent);
        });
    }
}