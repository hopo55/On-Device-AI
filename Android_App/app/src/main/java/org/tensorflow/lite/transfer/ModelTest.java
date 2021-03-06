package org.tensorflow.lite.transfer;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.drawable.Drawable;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Handler;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import static android.speech.tts.TextToSpeech.ERROR;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.transfer.api.AssetModelLoader;
import org.tensorflow.lite.transfer.api.TransferLearningModel;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;

public class ModelTest extends AppCompatActivity implements SensorEventListener {
    private static final String TAG = "TF_Lite";
    public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";

    private int in_nc = 6;
    private int segmentSize = 250;
    private int limit = 50;

    private int sampling_rate = 10000;  // 100Hz(0.01s)
    private SensorManager sensorManager;
    private Sensor acc_sensor;
    private Sensor gyro_sensor;

    private String mModuleAssetName;
    private TransferLearningModel model;
    private String[] label = {"0", "1", "2", "3"};
    private String[] view_label = {"Up", "Down", "Right Circle", "Left Circle"};
    private long MillisecondTime, StartTime, TimeBuff, UpdateTime = 0L;
    private int Seconds, Minutes;

    private Handler handler;

    private FileInputStream fileInputStream_train;
    private CSVReader cr_train;
    private String fileName_saveModel;
    private ByteBuffer[] trainParameters;

    private TextView timerView;
    private TextView result_text;
    private Button model_run_button;

    private TextToSpeech tts;
    private boolean RUN;
    private boolean timer_flag;
    private boolean value_check;

    private Drawable[] drawables = new Drawable[4];


    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.train_model_run);

        mModuleAssetName = getIntent().getStringExtra(INTENT_MODULE_ASSET_NAME);
        fileName_saveModel = mModuleAssetName + "_TrainModel.csv";

        String path_data = getFilesDir().getAbsolutePath() + "/" + fileName_saveModel;
        File file_train = new File(path_data);
        RUN = true;

        timerView = findViewById(R.id.test_timerView);
        result_text = findViewById(R.id.test_resultView);
        model_run_button = findViewById(R.id.train_model_run);

        handler = new Handler();

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        acc_sensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        gyro_sensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

//        for (int i=0; i<label.length; i++) {
//            String target_name = "target_" + label[i];
//            drawables[i] = getApplicationContext().getResources().getIdentifier(target_name, "drawable", getPackageName()));
//        }

        if (file_train.exists()) {
            model_run_button.setOnClickListener(v -> {
                if (RUN) {
                    RUN = false;
                    model_run_button.setText("MODEL STOP");
                    try {
                        timer_flag = false;

                        fileInputStream_train = openFileInput(fileName_saveModel);
                        cr_train = new CSVReader(new InputStreamReader(fileInputStream_train));
                        List<String[]> nextLine_train = cr_train.readAll();
                        trainParameters = new ByteBuffer[nextLine_train.size()];

                        for (int i=0; i<nextLine_train.size(); i++) {
                            trainParameters[i] = allocateBuffer(nextLine_train.get(i).length);
                            for (int n=0; n<nextLine_train.get(i).length; n++) {
                                trainParameters[i].put(Byte.parseByte(nextLine_train.get(i)[n]));
                            }
                        }
                        for (ByteBuffer buffer : trainParameters) {
                            buffer.rewind();
                        }
                        cr_train.close();
                        fileInputStream_train.close();

                        sensorManager.registerListener(this, acc_sensor, sampling_rate);
                        sensorManager.registerListener(this, gyro_sensor, sampling_rate);

                        StartTime = SystemClock.uptimeMillis();
                        handler.postDelayed(runnable, 0);
                    } catch (CsvException e) {
                        e.printStackTrace();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                else {
                    RUN = true;
                    model_run_button.setText("MODEL RUN");
                    resetTimer();
                }
            });
        } else {
            Toast.makeText(getApplicationContext(), "????????? ????????? ????????????.", Toast.LENGTH_SHORT).show();
        }


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
        if(event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
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
    String result_label;
    ArrayList<Float> sensor_values;
    ArrayList<ArrayList> sensor_list = new ArrayList<>();

    public void put_data(String[] sensor_data){
        ArrayList<String> temp = new ArrayList<>();
        for (int i=0; i<sensor_data.length; i++) {
            temp.add(sensor_data[i]);
        }
        sensor_list.add(temp);

        if (timer_flag){
            preprocessing(sensor_list);
            if (value_check) {
                result_label = model_run(sensor_values);

//                result_text.setText("Result : " + result_label);
//                tts.speak(result_label, TextToSpeech.QUEUE_FLUSH, null, null);
            }
            else {
                result_text.setText("Input value error");
                tts.speak("Input value error", TextToSpeech.QUEUE_FLUSH, null, null);
            }
            onPause();
            FinishRecord();

            sensor_length = 0;
            sensor_list = new ArrayList<>();
        }
    }

    boolean flag = false;
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
            flag = true;
            value_check = true;
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

    public String model_run(ArrayList<Float> data) {
        int start = 0;
        int end = in_nc * segmentSize;
        float[] input;
        float[] input_temp = new float[data.size()];

        for (int i=0; i<data.size(); i++) {
            input_temp[i] = data.get(i);
        }

        if (flag) {
            input = Arrays.copyOfRange(input_temp, start, end);
            model = new TransferLearningModel(new AssetModelLoader(this, mModuleAssetName), label);

            TransferLearningModel.Prediction[] predictions = model.predictTrain(input, trainParameters);
            result_text.setText(view_label[Integer.parseInt(predictions[0].getClassName())]);
            int draw_idx = Integer.parseInt(predictions[0].getClassName()) + 5;
            String draw_name = "target_" + draw_idx;
            result_text.setCompoundDrawablesWithIntrinsicBounds(getResources().getIdentifier(draw_name, "drawable", getPackageName()), 0, 0, 0);
            tts.speak(view_label[Integer.parseInt(predictions[0].getClassName())], TextToSpeech.QUEUE_FLUSH, null, null);

            return view_label[Integer.parseInt(predictions[0].getClassName())];
        }
        else {
            result_text.setText("Input Error");
            tts.speak("Input Error", TextToSpeech.QUEUE_FLUSH, null, null);
            return "None";
        }
    }

    private static ByteBuffer allocateBuffer(int capacity) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(capacity);
        buffer.order(ByteOrder.nativeOrder());

        return buffer;
    }

    public void FinishRecord() {
        CharSequence text ="Finish Model Run";
        int duration = Toast.LENGTH_SHORT;
        Toast toast = Toast.makeText(this, text, duration);
        toast.show();
    }

    @Override
    protected void onStop() {
        super.onStop();
        if (tts != null) {
            tts.stop();
            tts.shutdown();
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

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
//                    Log.i(TAG, "OpenCV loaded successfully");
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
