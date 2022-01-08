package org.tensorflow.lite.transfer;

import android.content.Intent;
import android.os.Bundle;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public class TransferList extends AppCompatActivity {
    public static final String INTENT_MODULE_NAME = "INTENT_MODE_NAME";
    private String moduleName;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.transfer_list);

        moduleName = getIntent().getStringExtra(INTENT_MODULE_NAME);

        findViewById(R.id.transfer_learning).setOnClickListener(v -> {
            final Intent intent = new Intent(TransferList.this, TrainData.class);
            intent.putExtra(TrainData.INTENT_MODULE_ASSET_NAME, moduleName);
            startActivity(intent);
        });

        findViewById(R.id.transfer_learning_test).setOnClickListener(v -> {
            final Intent intent = new Intent(TransferList.this, ModelTest.class);
            intent.putExtra(ModelTest.INTENT_MODULE_ASSET_NAME, moduleName);
            startActivity(intent);
        });

        findViewById(R.id.tl_auto_test).setOnClickListener(v -> {
            final Intent intent = new Intent(TransferList.this, ModelTrainTest.class);
            intent.putExtra(ModelTrainTest.INTENT_MODULE_ASSET_NAME, moduleName);
            startActivity(intent);
        });
    }
}
