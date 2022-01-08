package org.tensorflow.lite.transfer;

import android.app.ActivityManager;
import android.app.ActivityManager.MemoryInfo;
import android.content.Context;
import android.os.Bundle;
import android.os.ConditionVariable;
import android.speech.tts.TextToSpeech;
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
import org.tensorflow.lite.transfer.api.AssetModelLoader;
import org.tensorflow.lite.transfer.api.TransferLearningModel;
import org.w3c.dom.Text;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;

import static android.speech.tts.TextToSpeech.ERROR;

public class ModelTrainTest extends AppCompatActivity {
    private static final String TAG = "TF_Lite";
    public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";

    private int in_nc = 6;
    private int segmentSize = 250;
    private int class_num = 4;
    private int limit = 1;
    private String mModuleAssetName;
    private String[] label = {"5", "6", "7", "8"};
    private String[] train_label_list = {"0", "1", "2", "3"};
    private String[] view_label = {"Target5", "Target6", "Target7", "Target8"};

    private TransferLearningModel model;
    private volatile TransferLearningModel.LossConsumer lossConsumer;

    private FileInputStream fileInputStream_train;
    private CSVReader cr_train;
    private FileInputStream fileInputStream_data;
    private CSVReader cr_data;
    private FileInputStream fileInputStream_label;
    private CSVReader cr_label;
    private String fileName_saveModel;
    private String fileName_saveLoss;
    private String fileName_cl_log;
    private String fileName_rh_log;

    private String fileName_test_data = "TestData.csv";
    private String fileName_test_label = "TestLabel.csv";
    private String fileName_train_data = "tl_TrainData.csv";
    private String fileName_train_label = "tl_TrainLabel.csv";
    private String[] cl_data_list = {"cl_TrainData_4.csv", "cl_TrainData_5.csv", "cl_TrainData_6.csv", "cl_TrainData_7.csv"};
    private String[] cl_label_list = {"cl_TrainLabel_4.csv", "cl_TrainLabel_5.csv", "cl_TrainLabel_6.csv", "cl_TrainLabel_7.csv"};
    private String[] rh_data_list = {"rh_TrainData_4.csv", "rh_TrainData_5.csv", "rh_TrainData_6.csv", "rh_TrainData_7.csv"};
    private String[] rh_label_list = {"rh_TrainLabel_4.csv", "rh_TrainLabel_5.csv", "rh_TrainLabel_6.csv", "rh_TrainLabel_7.csv"};

    private ByteBuffer[] trainParameters;

    private ByteBuffer[] modelParameters;
    private FileOutputStream fileOutputStream_saveModel;
    private CSVWriter cw_saveModel;

    private FileOutputStream fileOutputStream_saveLoss;
    private CSVWriter cw_saveLoss;

    private FileInputStream fileInputStream_saveLog;
    private CSVReader cr_saveLog;
    private FileOutputStream fileOutputStream_saveLog;
    private CSVWriter cw_saveLog;
    private ArrayList<ArrayList> cl_log;
    private ArrayList<ArrayList> rh_log;
    private File file_cl_log;
    private File file_rh_log;
    private int cl_count;
    private int rh_count;
    private int break_point;


    private final ConditionVariable shouldTrain = new ConditionVariable();
    private MutableLiveData<Float> lastLoss = new MutableLiveData<>();
    private ArrayList loss_list = new ArrayList();

    private TextView[] textViews;
    private TextView result_text;
    private TextView model_inference_button;
    private TextView tl_model_train_button;
    private TextView tl_model_stop_button;
    private TextView tl_model_test_button;
    private TextView cl_train_count_text;
    private TextView cl_model_train_button;
    private TextView cl_model_stop_button;
    private TextView cl_model_test_button;
    private TextView cl_rehearsal_button;

    private TextToSpeech tts;
    private boolean inf_flag;
    private boolean tl_flag;
    private boolean cl_flag;
    private boolean rh_flag;
    private boolean[] cl_flag_list = new boolean[class_num];

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.auto_test);

        mModuleAssetName = getIntent().getStringExtra(INTENT_MODULE_ASSET_NAME);
        fileName_saveLoss = mModuleAssetName + "_LossValues.csv";

        String path_train_data = getFilesDir().getAbsolutePath() + "/" + fileName_train_data;
        String path_train_label = getFilesDir().getAbsolutePath() + "/" + fileName_train_label;
        String path_test_data = getFilesDir().getAbsolutePath() + "/" + fileName_test_data;
        String path_test_label = getFilesDir().getAbsolutePath() + "/" + fileName_test_label;

        File file_train_data = new File(path_train_data);
        File file_train_label = new File(path_train_label);
        File file_test_data = new File(path_test_data);
        File file_test_label = new File(path_test_label);

        break_point = 0;
        cl_count = 0;
        rh_count = 0;
        cl_log = new ArrayList<>();
        rh_log = new ArrayList<>();

        inf_flag = true;
        tl_flag = true;
        cl_flag = true;
        rh_flag = true;

        result_text = findViewById(R.id.loss_textView);
//        model_inference_button = findViewById(R.idmodel_inference);
//        tl_model_train_button = findViewById(R.id.tl_train_button);
//        tl_model_stop_button = findViewById(R.id.tl_stop_button);
//        tl_model_test_button = findViewById(R.id.tl_test_button);
//        cl_train_count_text = findViewById(R.id.cl_train_count);
//        cl_model_train_button = findViewById(R.id.cl_train_button);
//        cl_model_stop_button = findViewById(R.id.cl_stop_button);
//        cl_model_test_button = findViewById(R.id.cl_test_button);
//        cl_rehearsal_button = findViewById(R.id.cl_rehearsal_button);

        textViews = new TextView[class_num];
        for (int i=0; i<textViews.length; i++) {
            String textViewId = "test_" + view_label[i];
            textViews[i] = findViewById(getResources().getIdentifier(textViewId, "id", getPackageName()));

            cl_flag_list[i] = true;
        }

        fileName_cl_log = mModuleAssetName + "_CL_Log.csv";
        String path_cl_log = getFilesDir().getAbsolutePath() + "/" + fileName_cl_log;
        file_cl_log = new File(path_cl_log);

        fileName_rh_log = mModuleAssetName + "_RH_Log.csv";
        String path_rh_log = getFilesDir().getAbsolutePath() + "/" + fileName_rh_log;
        file_rh_log = new File(path_rh_log);

        if (file_cl_log.exists()) {
            try {
                fileInputStream_saveLog = openFileInput(fileName_cl_log);
                cr_saveLog = new CSVReader(new InputStreamReader(fileInputStream_saveLog));
                List<String[]> nextLine_log = cr_saveLog.readAll();
                cl_log = new ArrayList<>();

                for (int i = 0; i < nextLine_log.size(); i++) {
                    ArrayList cl_log_temp = new ArrayList();
                    for (int n = 0; n < nextLine_log.get(i).length; n++) {
                        cl_log_temp.add(nextLine_log.get(i)[n]);
                    }
                    cl_log.add(cl_log_temp);
                }
                cr_saveLog.close();
                fileInputStream_saveLog.close();

                int last_idx = cl_log.size() - 1;
                cl_count = Integer.parseInt(cl_log.get(last_idx).get(0).toString());
                cl_train_count_text.setText("CL Training Count : " + cl_count);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } catch (CsvException e) {
                e.printStackTrace();
            }
        }

        if (file_rh_log.exists()) {
            try {
                fileInputStream_saveLog = openFileInput(fileName_rh_log);
                cr_saveLog = new CSVReader(new InputStreamReader(fileInputStream_saveLog));
                List<String[]> nextLine_log = cr_saveLog.readAll();
                rh_log = new ArrayList<>();

                for (int i = 0; i < nextLine_log.size(); i++) {
                    ArrayList rh_log_temp = new ArrayList();
                    for (int n = 0; n < nextLine_log.get(i).length; n++) {
                        rh_log_temp.add(nextLine_log.get(i)[n]);
                    }
                    rh_log.add(rh_log_temp);
                }
                cr_saveLog.close();
                fileInputStream_saveLog.close();

                int last_idx = rh_log.size() - 1;
                rh_count = Integer.parseInt(rh_log.get(last_idx).get(0).toString());
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } catch (CsvException e) {
                e.printStackTrace();
            }
        }

        model_inference_button.setOnClickListener(v -> {
            if (file_test_data.exists() && file_test_label.exists()) {
                ArrayList<ArrayList> inference_file = getData(fileName_test_data, fileName_test_label);
                ArrayList<ArrayList> inference_data = inference_file.get(0);
                ArrayList inference_label = inference_file.get(1);

                if (inf_flag) {
                    Toast.makeText(getApplicationContext(), "Start Inference", Toast.LENGTH_SHORT).show();

                    clickdisable();
                    model_inference_button.setText("Inference is Running...");

                    float acc = model_run(inference_data, inference_label, false) * 100;
                    result_text.setText("Average Accuracy : " + String.format("%.2f", acc) + "%");

                    clickable();
                    model_inference_button.setText("Model Inference");
                }
            }
        });

        tl_model_train_button.setOnClickListener(v -> {
            if (file_train_data.exists() && file_train_label.exists()) {
                ArrayList<ArrayList> train_file = getData(fileName_train_data, fileName_train_label);
                ArrayList<ArrayList> train_data = train_file.get(0);
                ArrayList train_label = train_file.get(1);

                if (train_data.size() > limit*label.length) {
                    if (tl_flag) {
                        break_point = 0;
                        model = new TransferLearningModel(new AssetModelLoader(this, mModuleAssetName), train_label_list);

                        Toast.makeText(getApplicationContext(), "Start Training", Toast.LENGTH_SHORT).show();

                        clickdisable();
                        tl_model_train_button.setText("Model is Running...");
                        model.addSample(train_data, train_label);
                        trainModel();       // train model
                        enableTraining((epoch, loss) -> setLastLoss(loss, result_text, "TL", tl_model_train_button));
                    }
                    else Toast.makeText(getApplicationContext(), "학습이 진행 중입니다.", Toast.LENGTH_SHORT).show();
                }
                else Toast.makeText(getApplicationContext(), "학습 데이터 수가 부족합니다.", Toast.LENGTH_SHORT).show();

            } else Toast.makeText(getApplicationContext(), "학습 데이터 또는 모델이 없습니다.", Toast.LENGTH_SHORT).show();

        });

        tl_model_stop_button.setOnClickListener(v -> {
            if (tl_flag == false) {
                disableTraining("TL", tl_model_train_button);
                Toast.makeText(getApplicationContext(), "학습을 멈췄습니다.", Toast.LENGTH_SHORT).show();
            }
            else Toast.makeText(getApplicationContext(), "학습중이 아닙니다.", Toast.LENGTH_SHORT).show();
        });

        tl_model_test_button.setOnClickListener(v -> {
            fileName_saveModel = "TL_" + mModuleAssetName + "_TrainModel.csv";
            String path_model = getFilesDir().getAbsolutePath() + "/" + fileName_saveModel;
            File file_model = new File(path_model);

            if (file_test_data.exists() && file_test_label.exists() && file_model.exists() && tl_flag) {
                Toast.makeText(getApplicationContext(), "Start Test", Toast.LENGTH_SHORT).show();
                tl_model_test_button.setText("Model Test is Running...");

                clickdisable();

                ArrayList<ArrayList> test_file = getData(fileName_test_data, fileName_test_label);
                ArrayList<ArrayList> test_data = test_file.get(0);
                ArrayList test_label = test_file.get(1);

                try {
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
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                } catch (CsvException e) {
                    e.printStackTrace();
                }

                float acc = model_run(test_data, test_label, true) * 100;
                result_text.setText("Average Accuracy : " + String.format("%.2f", acc) + "%");
                tl_model_test_button.setText("TL Model Test");

                clickable();
            }
            else Toast.makeText(getApplicationContext(), "테스트 데이터 또는 모델이 없습니다.", Toast.LENGTH_SHORT).show();
        });

        for (int i=0; i<class_num; i++) {
            int idx = i;
            textViews[i].setOnClickListener(v -> {
                if (cl_flag_list[idx] == true) {
                    cl_flag_list[idx] = false;
                    textViews[idx].setBackgroundResource(R.drawable.edge_on);
                }
                else {
                    cl_flag_list[idx] = true;
                    textViews[idx].setBackgroundResource(R.drawable.edge);
                }
            });
        }

        cl_model_train_button.setOnClickListener(v -> {
            ArrayList<ArrayList> total_train_data = new ArrayList<>();
            ArrayList total_train_label = new ArrayList();

            for (int n=0; n<class_num; n++) {
                if (cl_flag_list[n] == false) {
                    ArrayList<ArrayList> train_file = getData(cl_data_list[n], cl_label_list[n]);
                    ArrayList<ArrayList> train_data = train_file.get(0);
                    ArrayList train_label = train_file.get(1);

                    for (int idx=0; idx<train_data.size(); idx++) {
                        total_train_data.add(train_data.get(idx));
                        total_train_label.add(train_label.get(idx));
                    }
                }
            }

            if (rh_flag == false && rh_count != 0) {
                int last_idx = rh_count - 1;

                for (int i=1; i<rh_log.get(last_idx).size(); i++) {
                    ArrayList<String>  labels = new ArrayList<>(Arrays.asList(view_label));
                    int index = labels.indexOf(rh_log.get(last_idx).get(i).toString());
                    ArrayList<ArrayList> train_file = getData(rh_data_list[index], rh_label_list[index]);
                    ArrayList<ArrayList> train_data = train_file.get(0);
                    ArrayList train_label = train_file.get(1);

                    for (int idx=0; idx<train_data.size(); idx++) {
                        total_train_data.add(train_data.get(idx));
                        total_train_label.add(train_label.get(idx));
                    }
                }
            }
            Log.d(TAG, "total_train_data : " + total_train_label.size());

            if (total_train_data.isEmpty() == false) {
                if (total_train_data.size() > limit*label.length) {
                    if (cl_flag) {
                        break_point = 0;
                        model = new TransferLearningModel(new AssetModelLoader(this, mModuleAssetName), train_label_list);
                        Toast.makeText(getApplicationContext(), "Start Training", Toast.LENGTH_SHORT).show();

                        clickdisable();
                        cl_model_train_button.setText("Model is Running...");
                        model.addSample(total_train_data, total_train_label);
                        trainModel();       // train model
                        enableTraining((epoch, loss) -> setLastLoss(loss, result_text, "CL", cl_model_train_button));
                    }
                    else Toast.makeText(getApplicationContext(), "학습이 진행 중입니다.", Toast.LENGTH_SHORT).show();
                }
                else Toast.makeText(getApplicationContext(), "학습 데이터 수가 부족합니다.", Toast.LENGTH_SHORT).show();
            }
            else Toast.makeText(getApplicationContext(), "학습 데이터가 없습니다.", Toast.LENGTH_SHORT).show();
        });

        cl_model_stop_button.setOnClickListener(v -> {
            if (cl_flag == false) {
                disableTraining("CL", cl_model_train_button);
                Toast.makeText(getApplicationContext(), "학습을 멈췄습니다.", Toast.LENGTH_SHORT).show();
            }
            else Toast.makeText(getApplicationContext(), "학습중이 아닙니다.", Toast.LENGTH_SHORT).show();
        });

        cl_model_test_button.setOnClickListener(v -> {
            fileName_saveModel = "CL_" + mModuleAssetName + "_TrainModel.csv";
            String path_model = getFilesDir().getAbsolutePath() + "/" + fileName_saveModel;
            File file_model = new File(path_model);

            if (file_test_data.exists() && file_test_label.exists() && file_model.exists() && cl_flag) {
                Toast.makeText(getApplicationContext(), "Start Test", Toast.LENGTH_SHORT).show();
                cl_model_test_button.setText("Model Test is Running...");

                clickdisable();

                ArrayList<ArrayList> test_file = getData(fileName_test_data, fileName_test_label);
                ArrayList<ArrayList> test_data = test_file.get(0);
                ArrayList test_label = test_file.get(1);

                try {
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
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                } catch (CsvException e) {
                    e.printStackTrace();
                }

                float acc = model_run(test_data, test_label, true) * 100;
                result_text.setText("Average Accuracy : " + String.format("%.2f", acc) + "%");
                cl_model_test_button.setText("CL Model Test");

                clickable();
            }
            else Toast.makeText(getApplicationContext(), "테스트 데이터 또는 모델이 없습니다.", Toast.LENGTH_SHORT).show();
        });

        cl_rehearsal_button.setOnClickListener(V -> {
            if (rh_flag == true) {
                rh_flag = false;
                cl_rehearsal_button.setBackgroundResource(R.drawable.edge_on);

                cl_train_count_text.setText("RH Training Count : " + rh_count);
            }
            else {
                rh_flag = true;
                cl_rehearsal_button.setBackgroundResource(R.drawable.edge);

                cl_train_count_text.setText("CL Training Count : " + cl_count);
            }
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

    public ArrayList<ArrayList> getData(String path_data, String path_label) {
        ArrayList file_temp;
        ArrayList<ArrayList> file_data = new ArrayList<>();
        ArrayList file_label = new ArrayList<>();
        ArrayList<ArrayList> total_file = new ArrayList<>();

        try {
            fileInputStream_data = openFileInput(path_data);
            cr_data = new CSVReader(new InputStreamReader(fileInputStream_data));
            fileInputStream_label = openFileInput(path_label);
            cr_label = new CSVReader(new InputStreamReader(fileInputStream_label));

            String[] nextLine_data;
            while ((nextLine_data = cr_data.readNext()) != null) {
                file_temp = new ArrayList();
                for (int i=0; i<nextLine_data.length; i++) {
                    file_temp.add(nextLine_data[i]);
                }
                file_data.add(file_temp);
            }

            String[] nextLine_label;
            while ((nextLine_label = cr_label.readNext()) != null) {
                file_label.add(nextLine_label[0]);
            }

            total_file.add(file_data);
            total_file.add(file_label);

            cr_data.close();
            fileInputStream_data.close();
            cr_label.close();
            fileInputStream_label.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (CsvValidationException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return total_file;
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
        loss_list.add(newLoss);

        new Thread(new Runnable() {
            @Override
            public void run() {
                runOnUiThread(new Runnable() {
                    public void run() {
                        lossView.setText("Training Loss : " + newLoss);
                        if (newLoss < 5e-4 && break_point == 0) {
                            break_point = 1;
                            disableTraining(mode, mode_button);
                        }
                    }
                });
            }
        }).start();
    }

    private void disableTraining(String mode, TextView mode_button) {
        clickable();

        shouldTrain.close();
        mode_button.setText(mode + " Model Train");
        result_text.setText("Finish Loss : " + lastLoss.getValue());
        Toast.makeText(getApplicationContext(), "End of Training", Toast.LENGTH_SHORT).show();

        try {
            fileName_saveModel = mModuleAssetName + "_TrainModel.csv";
            fileOutputStream_saveModel = openFileOutput(mode + "_" + fileName_saveModel, Context.MODE_PRIVATE);
            cw_saveModel = new CSVWriter(new OutputStreamWriter(fileOutputStream_saveModel));

            modelParameters = model.saveModelParameters();

            for (int i=0; i<modelParameters.length; i++) {
                String[] data = new String[modelParameters[i].limit()];
                for (int n=0; n<modelParameters[i].limit(); n++) {
                    data[n] = String.valueOf(modelParameters[i].get(n));
                }
                cw_saveModel.writeNext(data);
            }
            cw_saveModel.close();
            fileOutputStream_saveModel.close();
            // Save Loss
            fileOutputStream_saveLoss = openFileOutput(fileName_saveLoss, Context.MODE_PRIVATE);
            cw_saveLoss = new CSVWriter(new OutputStreamWriter(fileOutputStream_saveLoss));

            String[] loss_value = new String[loss_list.size()];
            for (int i=0; i<loss_list.size(); i++) {
                loss_value[i] = loss_list.get(i).toString();
            }
            cw_saveLoss.writeNext(loss_value);

            cw_saveLoss.close();
            fileOutputStream_saveLoss.close();
            // Save Log
            if (mode == "CL") {
                if (rh_flag) {
                    rec_log(file_cl_log, fileName_cl_log, cl_log, "CL");
                }
                else {
                    rec_log(file_rh_log, fileName_rh_log, rh_log, "RH");
                }
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void rec_log(File log_file, String log_fileName, ArrayList<ArrayList> log_array, String mode) {
        try {
            if (log_file.exists() & log_array.size() != 0) {
                fileOutputStream_saveLog = openFileOutput(log_fileName, Context.MODE_APPEND);
                cw_saveLog = new CSVWriter(new OutputStreamWriter(fileOutputStream_saveLog));
                ArrayList log_list = new ArrayList();
                int last_idx = log_array.size() - 1;
                int add_idx = Integer.parseInt(log_array.get(last_idx).get(0).toString()) + 1;

                log_list.add(add_idx);

                if (mode == "RH") {
                    int rh_last_idx = rh_log.size() - 1;
                    for (int i=1; i<rh_log.get(rh_last_idx).size(); i++) {
                        log_list.add(rh_log.get(rh_last_idx).get(i));
                    }
                }

                for (int i=0; i<cl_flag_list.length; i++) {
                    if (cl_flag_list[i] == false) log_list.add(view_label[i]);
                }

                String[] log_temp = new String[log_list.size()];
                for (int i=0; i<log_list.size(); i++) {
                    log_temp[i] = log_list.get(i).toString();
                }
                cw_saveLog.writeNext(log_temp);

                log_array.add(log_list);
                if (mode == "RH") {
                    rh_count = Integer.parseInt(log_array.get(last_idx+1).get(0).toString());
                    cl_train_count_text.setText("RH Training Count : " + rh_count);
                }
                else {
                    cl_count = Integer.parseInt(log_array.get(last_idx+1).get(0).toString());
                    cl_train_count_text.setText("CL Training Count : " + cl_count);
                }
            }
            else {
                fileOutputStream_saveLog = openFileOutput(log_fileName, Context.MODE_PRIVATE);
                cw_saveLog = new CSVWriter(new OutputStreamWriter(fileOutputStream_saveLog));
                ArrayList log_list = new ArrayList();

                log_list.add("1");
                for (int i=0; i<cl_flag_list.length; i++) {
                    if (cl_flag_list[i] == false) log_list.add(view_label[i]);
                }

                String[] log_temp = new String[log_list.size()];
                for (int i=0; i<log_list.size(); i++) {
                    log_temp[i] = log_list.get(i).toString();
                }
                cw_saveLog.writeNext(log_temp);

                log_array.add(log_list);
                if (mode == "RH") {
                    rh_count = Integer.parseInt(log_array.get(0).get(0).toString());
                    cl_train_count_text.setText("RH Training Count : " + rh_count);
                }
                else {
                    cl_count = Integer.parseInt(log_array.get(0).get(0).toString());
                    cl_train_count_text.setText("CL Training Count : " + cl_count);
                }
            }
            cw_saveLog.close();
            fileOutputStream_saveLog.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
//        catch (CsvException e) {
//            e.printStackTrace();
//        }
    }

    public float model_run(ArrayList<ArrayList> input_data, ArrayList target_label, boolean tl_model) {
        float accuracy = 0;
        int[][] result_list = new int[2][class_num];
        model = new TransferLearningModel(new AssetModelLoader(this, mModuleAssetName), train_label_list);

        for (int i=0; i<input_data.size(); i++) {
            float[] input = new float[in_nc*segmentSize];
            result_list[1][Integer.parseInt(target_label.get(i).toString())] += 1;
            for (int n=0; n<in_nc*segmentSize; n++) input[n] = Float.parseFloat(input_data.get(i).get(n).toString());

            TransferLearningModel.Prediction[] predictions;
            if (tl_model) predictions = model.predictTrain(input, trainParameters);
            else predictions = model.predict(input);

            if (target_label.get(i).equals(predictions[0].getClassName())) {
                accuracy += 1;
                result_list[0][Integer.parseInt(target_label.get(i).toString())] += 1;
            }
        }

        for (int i=0; i<class_num; i++) {
            String result = view_label[i] + "\n" + result_list[0][i] + " / " + result_list[1][i];
            float acc = ((float)result_list[0][i] / (float)result_list[1][i]) * 100;
            textViews[i].setText(result + " (" + String.format("%.2f", acc) + "%)");
        }

        accuracy /= input_data.size();

        return accuracy;
    }

    private static ByteBuffer allocateBuffer(int capacity) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(capacity);
        buffer.order(ByteOrder.nativeOrder());

        return buffer;
    }

    public void clickable() {
        inf_flag = true;
        tl_flag = true;
        cl_flag = true;
    }

    public void clickdisable() {
        inf_flag = false;
        tl_flag = false;
        cl_flag = false;
    }

    @Override
    protected void onStop() {
        super.onStop();
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
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
