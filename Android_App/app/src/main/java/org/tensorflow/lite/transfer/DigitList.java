package org.tensorflow.lite.transfer;

import android.content.Intent;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;

public class DigitList extends AppCompatActivity {
    public static final String INTENT_SAVE_TL_NAME = "None";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.digit_list);
        String saveTLName = getIntent().getStringExtra(INTENT_SAVE_TL_NAME);

        findViewById(R.id.target_1).setOnClickListener(v -> {
            final Intent intent = new Intent(DigitList.this, CollectData.class);
            intent.putExtra(CollectData.INTENT_SAVE_NAME, "Target1_");
            intent.putExtra(CollectData.INTENT_SAVE_TL_NAME, saveTLName + "_");
            startActivity(intent);
        });

        findViewById(R.id.target_2).setOnClickListener(v -> {
            final Intent intent = new Intent(DigitList.this, CollectData.class);
            intent.putExtra(CollectData.INTENT_SAVE_NAME, "Target2_");
            intent.putExtra(CollectData.INTENT_SAVE_TL_NAME, saveTLName + "_");
            startActivity(intent);
        });

        findViewById(R.id.target_3).setOnClickListener(v -> {
            final Intent intent = new Intent(DigitList.this, CollectData.class);
            intent.putExtra(CollectData.INTENT_SAVE_NAME, "Target3_");
            intent.putExtra(CollectData.INTENT_SAVE_TL_NAME, saveTLName + "_");
            startActivity(intent);
        });

        findViewById(R.id.target_4).setOnClickListener(v -> {
            final Intent intent = new Intent(DigitList.this, CollectData.class);
            intent.putExtra(CollectData.INTENT_SAVE_NAME, "Target4_");
            intent.putExtra(CollectData.INTENT_SAVE_TL_NAME, saveTLName + "_");
            startActivity(intent);
        });

        findViewById(R.id.target_5).setOnClickListener(v -> {
            final Intent intent = new Intent(DigitList.this, CollectData.class);
            intent.putExtra(CollectData.INTENT_SAVE_NAME, "Target5_");
            intent.putExtra(CollectData.INTENT_SAVE_TL_NAME, saveTLName + "_");
            startActivity(intent);
        });

        findViewById(R.id.target_6).setOnClickListener(v -> {
            final Intent intent = new Intent(DigitList.this, CollectData.class);
            intent.putExtra(CollectData.INTENT_SAVE_NAME, "Target6_");
            intent.putExtra(CollectData.INTENT_SAVE_TL_NAME, saveTLName + "_");
            startActivity(intent);
        });

        findViewById(R.id.target_7).setOnClickListener(v -> {
            final Intent intent = new Intent(DigitList.this, CollectData.class);
            intent.putExtra(CollectData.INTENT_SAVE_NAME, "Target7_");
            intent.putExtra(CollectData.INTENT_SAVE_TL_NAME, saveTLName + "_");
            startActivity(intent);
        });

        findViewById(R.id.target_8).setOnClickListener(v -> {
            final Intent intent = new Intent(DigitList.this, CollectData.class);
            intent.putExtra(CollectData.INTENT_SAVE_NAME, "Target8_");
            intent.putExtra(CollectData.INTENT_SAVE_TL_NAME, saveTLName + "_");
            startActivity(intent);
        });
    }
}
