package org.tensorflow.lite.transfer;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Environment;
import android.os.Handler;
import android.os.ParcelFileDescriptor;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.documentfile.provider.DocumentFile;

import com.opencsv.CSVWriter;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

import static android.speech.tts.TextToSpeech.ERROR;

public class CollectData extends AppCompatActivity implements SensorEventListener {
    public static final String INTENT_SAVE_NAME = "INTENT_SAVE_NAME";
    public static final String INTENT_SAVE_TL_NAME = "None";

    private static final String TAG = "TL";

    private int in_nc = 6;
    private int segmentSize = 250;
    private int limit = 50;
    private String saveName;

    private SensorManager sensorManager;
    private Sensor acc_sensor;
    private Sensor gyro_sensor;

    private TextView collect_data;
    private TextView sampling_rate_text;
    private TextView timerView;
    private TextView accValue;
    private TextView gyroValue;

    private Handler handler;

    private boolean RUN;
    private boolean timer_flag;
    private boolean value_check;

    private long MillisecondTime, StartTime, TimeBuff, UpdateTime = 0L;
    private int Seconds, Minutes;

    private TextToSpeech tts;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.collect_data);

        RUN = true;

        handler = new Handler();

        collect_data = findViewById(R.id.collect_button);
        sampling_rate_text = findViewById(R.id.sampling);
        timerView = findViewById(R.id.timerView);
        accValue = findViewById(R.id.accValues);
        gyroValue = findViewById(R.id.gyroValues);

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        acc_sensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        gyro_sensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        saveName = getIntent().getStringExtra(INTENT_SAVE_NAME);

        SAF();

        collect_data.setOnClickListener(v -> {
            if  (RUN) {
                RUN = false;
                collect_data.setText("STOP");

                try {
                    final int sampling_rate = Integer.parseInt(sampling_rate_text.getText().toString());
                    timer_flag = false;

                    // 100,000ms = 0.1s, Normal = 200,000(0.2s), UI = 60,000(0.06s), Game = 20,000(0.02s), Fastest = 0  ms = microseconds
                    sensorManager.registerListener(this, acc_sensor, sampling_rate);
                    sensorManager.registerListener(this, gyro_sensor, sampling_rate);

                    StartTime = SystemClock.uptimeMillis();
                    handler.postDelayed(runnable, 0);
                } catch (Exception e){
                    CharSequence text ="Input value error";
                    int duration = Toast.LENGTH_SHORT;
                    Toast toast = Toast.makeText(CollectData.this, text, duration);
                    toast.show();
                }
            }
            else {
                RUN = true;
                collect_data.setText("COLLECT DATA");
                resetTimer();
            }
        });

        findViewById(R.id.collect_new_button).setOnClickListener(v -> {
            SAF();
        });

        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status != ERROR) {
                    tts.setLanguage(Locale.KOREAN);
                }
            }
        });
    }

    String[] values = new String[5];
    @Override
    public void onSensorChanged(SensorEvent event) {
        String x, y, z;
        if(event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION){
            x = String.format("%.2f", event.values[0]);
            y = String.format("%.2f", event.values[1]);
            z = String.format("%.2f", event.values[2]);
            accValue.setText("x : " + x + ", y : " + y + ", z : " + z);
            values[0] = "ACC"; values[1] = x; values[2] = y; values[3] = z; values[4] = String.valueOf(event.timestamp);
        } else if(event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            x = String.format("%.2f", event.values[0]);
            y = String.format("%.2f", event.values[1]);
            z = String.format("%.2f", event.values[2]);
            gyroValue.setText("x : " + x + ", y : " + y + ", z : " + z);
            values[0] = "GYRO"; values[1] = x; values[2] = y; values[3] = z; values[4] = String.valueOf(event.timestamp);
        }
        putdata(values);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Do something here if sensor accuracy changes.
    }

    @Override
    protected void onPause() {
        super.onPause();
        sensorManager.unregisterListener(this);
    }

    int sensor_length = 0;
    String[][] sensor_values = new String[segmentSize][in_nc];
    ArrayList<ArrayList> sensor_list = new ArrayList<>();

    public void putdata(String[] sensor_data) {
        ArrayList<String> temp = new ArrayList<>();
        for (int i=0; i<sensor_data.length; i++) {
            temp.add(sensor_data[i]);
        }
        sensor_list.add(temp);

        if (timer_flag){
            preprocessing(sensor_list);
            if (value_check && fileOutputStream!=null) {
                for (int i=0; i<segmentSize; i++) {
                    cw.writeNext(sensor_values[i]);
                }
                FinishRecord();

                Toast.makeText(getApplicationContext(), "Save Data", Toast.LENGTH_SHORT).show();
                tts.speak("Save Data", TextToSpeech.QUEUE_FLUSH, null, null);
            } else {
                Toast.makeText(getApplicationContext(), "Input Error", Toast.LENGTH_SHORT).show();
                tts.speak("Input Error", TextToSpeech.QUEUE_FLUSH, null, null);
            }
            onPause();

            sensor_length = 0;
            sensor_list = new ArrayList<>();
        }
    }

    ArrayList<ArrayList> data_list;

    public void preprocessing(ArrayList<ArrayList> sensor_data){
        value_check = false;

        data_list = new ArrayList<>();
        ArrayList<String> acc_temp;
        ArrayList<ArrayList> acc_data = new ArrayList<>();
        ArrayList<String> gyro_temp;
        ArrayList<ArrayList> gyro_data = new ArrayList<>();

        for (int i=0; i<sensor_data.size(); i++){
            if (sensor_data.get(i).get(0) == "ACC"){
                acc_temp = new ArrayList<>();
                for (int n=1; n<sensor_data.get(i).size(); n++) {
                    acc_temp.add(sensor_data.get(i).get(n).toString());
                }
                acc_data.add(acc_temp);
            } else if (sensor_data.get(i).get(0) == "GYRO") {
                gyro_temp = new ArrayList<>();
                for (int n=1; n<sensor_data.get(i).size(); n++) {
                    gyro_temp.add(sensor_data.get(i).get(n).toString());
                }
                gyro_data.add(gyro_temp);
            }
        }

        if (acc_data.size() >= limit && gyro_data.size() >= limit) {
            value_check = true;
            sensor_values = new String[segmentSize][in_nc];

            int col = 0;
            float[][] resize_acc = new float[3][acc_data.size()];
            float[][] resize_gyro = new float[3][gyro_data.size()];

            Mat original_acc = new Mat(3, acc_data.size(), CvType.CV_32F);
            Mat original_gyro = new Mat(3, gyro_data.size(), CvType.CV_32F);
            Mat resize = new Mat(6, segmentSize, CvType.CV_32F);

            for (int i=0; i<acc_data.size(); i++) {
                for (int j=0; j<3; j++) {
                    resize_acc[j][i] = Float.parseFloat(acc_data.get(i).get(j).toString());
                }
            }
            for (int i=0; i<gyro_data.size(); i++) {
                for (int j=0; j<3; j++) {
                    resize_gyro[j][i] = Float.parseFloat(gyro_data.get(i).get(j).toString());
                }
            }

            for (int i=0; i<3; i++) {
                original_acc.put(i, col, resize_acc[i]);
                Imgproc.resize(original_acc.row(i), resize.row(i), new Size(segmentSize, 1), 1, 1, Imgproc.INTER_CUBIC);
                original_gyro.put(i, col, resize_gyro[i]);
                Imgproc.resize(original_gyro.row(i), resize.row(i+3), new Size(segmentSize, 1), 1, 1, Imgproc.INTER_CUBIC);
            }

            for (int i=0; i<segmentSize; i++) {
                for (int j=0; j<in_nc; j++) {
                    double[] result = resize.get(j, i);
                    sensor_values[i][j] = String.valueOf(result[0]);
                }
            }
        }
    }

    public Runnable runnable = new Runnable() {
        public void run() {
            MillisecondTime = SystemClock.uptimeMillis() - StartTime;
            UpdateTime = TimeBuff + MillisecondTime;
            Seconds = (int) (UpdateTime / 1000);
            Minutes = Seconds / 60;
            Seconds = Seconds % 60;
            timerView.setText("" + String.format("%02d", Minutes) + ":"
                    + String.format("%02d", Seconds));

            handler.postDelayed(this, 0);
        }
    };

    public void resetTimer() {
        handler.removeCallbacks(runnable);

        MillisecondTime = 0L ;
        StartTime = 0L ;
        TimeBuff = 0L ;
        UpdateTime = 0L ;
        Seconds = 0 ;
        Minutes = 0 ;
        timer_flag = true;

        timerView.setText("00:00");
    }

    private int WRITE_REQUEST_CODE = 43;
    private ParcelFileDescriptor pfd;
    private FileOutputStream fileOutputStream;
    private CSVWriter cw;

    public void SAF() {
        try {
            long now = System.currentTimeMillis();
            Date date = new Date(now);
            @SuppressLint("SimpleDateFormat")SimpleDateFormat sDate = new SimpleDateFormat("yy_MM_dd_HH_mm_ss");
            String formatDate = sDate.format(date);
            String fileTitle = "";
            String saveTLName = getIntent().getStringExtra(INTENT_SAVE_TL_NAME);
            if (saveTLName.contains("TL_")){
                fileTitle = saveTLName + saveName + formatDate + ".csv";
            }
            else {
                fileTitle = saveName + formatDate + ".csv";
            }

            Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("*/*");
            intent.putExtra(Intent.EXTRA_TITLE, fileTitle);

            startActivityForResult(intent, WRITE_REQUEST_CODE);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == WRITE_REQUEST_CODE && resultCode == Activity.RESULT_OK && data != null){
            Uri uri = data.getData();
            addFile(uri);
        }
    }

    public void addFile(Uri uri){
        try {
            pfd = this.getContentResolver().openFileDescriptor(uri, "w");
            fileOutputStream = new FileOutputStream(pfd.getFileDescriptor());
            cw = new CSVWriter(new OutputStreamWriter(fileOutputStream));

        } catch (FileNotFoundException e){
            e.printStackTrace();
        }
    }

    public void FinishRecord() {
        try {
            cw.close();
            fileOutputStream.close();
            pfd.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }
}
