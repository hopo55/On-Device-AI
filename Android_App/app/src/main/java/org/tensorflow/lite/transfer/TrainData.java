package org.tensorflow.lite.transfer;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.ConditionVariable;
import android.os.Handler;
import android.os.SystemClock;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.MutableLiveData;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import com.opencsv.exceptions.CsvException;
import com.opencsv.exceptions.CsvValidationException;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.transfer.api.AssetModelLoader;
import org.tensorflow.lite.transfer.api.TransferLearningModel;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.ExecutionException;

public class TrainData extends AppCompatActivity implements SensorEventListener {
    private static final String TAG = "TF_Lite";
    public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";

    private int sampling_rate = 10000;  // 100Hz
    private int in_nc = 6;
    private int segmentSize = 250;
    private int limit = 50;
    private String[] view_label = {"Target1", "Target2", "Target3", "Target4"};

    private SensorManager sensorManager;
    private Sensor acc_sensor;
    private Sensor gyro_sensor;

    private TextView timerView;
    private TextView addView;

    private String label;
    private boolean timer_flag;
    private boolean flag;
    private boolean cl_flag;
    private boolean train_mode_flag;
    private boolean RUN;

    private Handler handler;

    private int Seconds, Minutes;
    private long MillisecondTime, StartTime, TimeBuff, UpdateTime = 0L;

    private HashMap<String, Integer> sample_count = new HashMap<>();
    private HashMap<String, String> label_count = new HashMap<>();
    private ArrayList<ArrayList> sample = new ArrayList();
    private ArrayList<ArrayList> cl_sample = new ArrayList();
    private ArrayList cl_sample_labels = new ArrayList();
    private ArrayList temp;
    private ArrayList sample_labels = new ArrayList();
    private ArrayList<ArrayList> transfer_data = new ArrayList<>();
    private ArrayList transfer_label = new ArrayList();

    private TextView cl_train_mode_button;
    private TextView cl_model_train_button;

    private int rh_count;
    private ArrayList<ArrayList> rh_sample = new ArrayList();
    private ArrayList rh_sample_labels = new ArrayList();

    private TransferLearningModel model;
    private String mModuleAssetName;
    private String[] train_label_list = {"0", "1", "2", "3"};
    private final ConditionVariable shouldTrain = new ConditionVariable();
    private volatile TransferLearningModel.LossConsumer lossConsumer;
    private MutableLiveData<Float> lastLoss = new MutableLiveData<>();
    private int break_point;
    private TextView result_text;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.transfer_learning);

        rh_count = 0;
        break_point = 0;
        cl_flag = true;

        RUN = true;
        train_mode_flag = false;
        timerView = findViewById(R.id.transfer_timerView);
        handler = new Handler();

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        acc_sensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        gyro_sensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        cl_train_mode_button = findViewById(R.id.train_mode_button);
        cl_model_train_button = findViewById(R.id.cl_train_mode_button);

        mModuleAssetName = getIntent().getStringExtra(INTENT_MODULE_ASSET_NAME);
        result_text = findViewById(R.id.loss_textView);

        TextView[] textViews = new TextView[4];
        for (int i=0; i<textViews.length; i++) {
            String textViewId = "test_" + view_label[i];
            textViews[i] = findViewById(getResources().getIdentifier(textViewId, "id", getPackageName()));

            sample_count.put(textViews[i].getText().toString(), 0);
            label_count.put(textViews[i].getText().toString(), String.valueOf(i));
        }

        cl_train_mode_button.setOnClickListener(v -> {
            if (train_mode_flag == false) {
                train_mode_flag = true;
                cl_train_mode_button.setBackgroundResource(R.drawable.edge_on);
            }
            else {
                train_mode_flag = false;
                cl_train_mode_button.setBackgroundResource(R.drawable.edge);
            }
        });

        for (TextView textViewId : textViews) {
            textViewId.setOnClickListener(v -> {
                if (RUN && train_mode_flag == false) {
                    RUN = false;
                    try {
                        timer_flag = false;
                        addView = textViewId;
                        addView.setBackgroundResource(R.drawable.edge_on);

                        sensorManager.registerListener(this, acc_sensor, sampling_rate);
                        sensorManager.registerListener(this, gyro_sensor, sampling_rate);

                        StartTime = SystemClock.uptimeMillis();
                        handler.postDelayed(runnable, 0);
                    } catch (Exception e) {
                        CharSequence text ="Input value error";
                        int duration = Toast.LENGTH_SHORT;
                        Toast toast = Toast.makeText(this, text, duration);
                        toast.show();
                    }
                }
                else if(RUN && train_mode_flag) {
                    RUN = false;
                    addView = textViewId;
                    addView.setBackgroundResource(R.drawable.train_mode_on);
                }
                else if (RUN == false && train_mode_flag) {
                    RUN = true;
                    addView.setBackgroundResource(R.drawable.edge);
                }
                else {
                    RUN = true;
                    addView.setBackgroundResource(R.drawable.edge);

                    resetTimer();
                }
            });
        }

        cl_model_train_button.setOnClickListener(v -> {
            if (RUN == false) {
                cl_sample = new ArrayList<>();
                cl_sample_labels = new ArrayList();

                for (int i=0; i<sample.size(); i++) {
                    String select_button = addView.getText().toString().split("\\s")[0];
                    String target_label = view_label[Integer.parseInt(sample_labels.get(i).toString())];
                    if (select_button.equals(target_label)) {
                        cl_sample.add(sample.get(i));
                        cl_sample_labels.add(sample_labels.get(i));
                    }
                }

                if (cl_sample.size() > 0) {
                    if (cl_flag) {
                        break_point = 0;
                        model = new TransferLearningModel(new AssetModelLoader(this, mModuleAssetName), train_label_list);
                        Toast.makeText(getApplicationContext(), "Start Training", Toast.LENGTH_SHORT).show();

                        cl_flag = false;
                        model.addSample(cl_sample, cl_sample_labels);
                        trainModel();       // train model
                        enableTraining((epoch, loss) -> setLastLoss(loss, result_text, "CL", cl_model_train_button));
                    }
                    else Toast.makeText(getApplicationContext(), "학습이 진행 중입니다.", Toast.LENGTH_SHORT).show();
                }
                else Toast.makeText(getApplicationContext(), "학습 데이터가 없습니다.", Toast.LENGTH_SHORT).show();
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
            values[0] = "ACC"; values[1] = x; values[2] = y; values[3] = z; values[4] = String.valueOf(event.timestamp);
        } else if(event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            x = String.format("%.2f", event.values[0]);
            y = String.format("%.2f", event.values[1]);
            z = String.format("%.2f", event.values[2]);
            values[0] = "GYRO"; values[1] = x; values[2] = y; values[3] = z; values[4] = String.valueOf(event.timestamp);
        }
        put_data(values);
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
    ArrayList<Float> sensor_values;
    ArrayList<ArrayList> sensor_list = new ArrayList<>();

    public void put_data(String[] sensor_data) {
        ArrayList<String> temp = new ArrayList<>();
        for (int i=0; i<sensor_data.length; i++) {
            temp.add(sensor_data[i]);
        }
        sensor_list.add(temp);

        if (timer_flag){
            preprocessing(sensor_list);
            if (flag) {
                changeView(addView);
                addSample(sensor_values);

                Toast.makeText(getApplicationContext(), "Success", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(getApplicationContext(), "Input Error", Toast.LENGTH_SHORT).show();
            }
            onPause();
            flag = false;
            sensor_length = 0;
            sensor_list = new ArrayList<>();
        }
    }

    ArrayList<ArrayList> data_list;

    public void preprocessing(ArrayList<ArrayList> sensor_data){
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
            flag = true;
            sensor_values = new ArrayList<>();

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
                    sensor_values.add((float) result[0]);
                }
            }
        }
    }

    public void addSample(ArrayList<Float> data) {
        int end = in_nc * segmentSize;

        temp = new ArrayList();
        for (int i=0; i<data.size(); i++){
            if (i == end) break;
            temp.add(data.get(i));
        }

        sample.add(temp);
        sample_labels.add(label);
    }

    public void FinishRecord() throws IOException {
        Toast.makeText(getApplicationContext(), "저장되었습니다.", Toast.LENGTH_SHORT).show();
    }

    public void changeView(TextView textViewId) {
        String sample_id = textViewId.getText().toString().split("\n")[0];
        Integer value = sample_count.get(sample_id) + 1;
        sample_count.put(sample_id, value);
        textViewId.setText(textViewId.getText().toString().split("\n")[0] + "\n" + sample_count.get(sample_id));

        label = label_count.get(sample_id);
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

    public void trainModel() {
        new Thread(() -> {
            while (!Thread.interrupted()) {
                shouldTrain.block();
                try {
                    model.train(1, lossConsumer).get();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }

    public void enableTraining(TransferLearningModel.LossConsumer lossConsumer) {
        this.lossConsumer = lossConsumer;
        shouldTrain.open();
    }

    public void setLastLoss(float newLoss, TextView lossView, String mode, TextView mode_button) {
        lastLoss.postValue(newLoss);

        new Thread(new Runnable() {
            @Override
            public void run() {
                runOnUiThread(new Runnable() {
                    public void run() {
                        lossView.setText("Training Loss : " + newLoss);
                        if (newLoss < 5e-4 && break_point == 0) {
                            break_point = 1;
                            cl_flag = true;
                            disableTraining(mode, mode_button);
                        }
                    }
                });
            }
        }).start();
    }

    private void disableTraining(String mode, TextView mode_button) {
        shouldTrain.close();
        mode_button.setText(mode + " Model Train");
        result_text.setText("Finish Loss : " + lastLoss.getValue());
        Toast.makeText(getApplicationContext(), "End of Training", Toast.LENGTH_SHORT).show();
    }
}
