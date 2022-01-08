package org.tensorflow.lite.transfer;

import android.content.Intent;
import android.os.Bundle;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public class TrainModelList extends AppCompatActivity {
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.model_list);

        findViewById(R.id.cnn2020_click_area).setOnClickListener(v -> {
            final Intent intent = new Intent(TrainModelList.this, TransferList.class);
            intent.putExtra(TransferList.INTENT_MODULE_NAME, "CNN2020");
            startActivity(intent);
        });

        findViewById(R.id.cnn_blstm_click_area).setOnClickListener(v -> {
            final Intent intent = new Intent(TrainModelList.this, TransferList.class);
            intent.putExtra(TransferList.INTENT_MODULE_NAME, "CNN_2Stream");
            startActivity(intent);
        });

        findViewById(R.id.innohar_click_area).setOnClickListener(v -> {
            final Intent intent = new Intent(TrainModelList.this, TransferList.class);
            intent.putExtra(TransferList.INTENT_MODULE_NAME, "InnoHAR");
            startActivity(intent);
        });
    }
}
