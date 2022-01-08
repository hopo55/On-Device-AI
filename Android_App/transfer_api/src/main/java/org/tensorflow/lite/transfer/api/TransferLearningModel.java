package org.tensorflow.lite.transfer.api;

import android.app.ActivityManager;
import android.content.Context;
import android.util.Log;
import android.widget.TextView;

import java.io.Closeable;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class TransferLearningModel implements Closeable {
    private static final String TAG = "TF_Lite";

    private static final int FLOAT_BYTES = 4;
//    private static final int numClasses = 16;
    private static final int numClasses = 4;
    private final String[] classesByIdx;
    private final int[] bottleneckShape;
    private final Map<String, Integer> classes;
    // Set to true when [close] has been called.
    // volatile -> Java 변수를 Main Memory에 저장하겠다는 것 -> CPU Cash에 저장
    private volatile boolean isTerminating = false;
    // This lock allows [close] method to assure that no threads are performing inference.
    // 동기화 시작시점과 끝점을 수동적으로 설정이 가능한 명시적 동기화
    private final Lock inferenceLock = new ReentrantLock();
    // This lock guards access to trainable parameters.
    private final ReadWriteLock parameterLock = new ReentrantReadWriteLock();

    private final LiteInitializeModel initializeModel;
    private final LiteInferenceModel inferenceModel;
    private final LiteTrainHeadModel trainHeadModel;
    private final LiteBottleneckModel bottleneckModel;
    private final LiteInferenceTrainModel inferenceTrainModel;
    private final LiteOptimizerModel optimizerModel;

    // Where to store training inputs.
    private final ByteBuffer trainingBatchBottlenecks;
    private final ByteBuffer trainingBatchClasses;
    // A zero-filled buffer of the same size as `trainingBatchClasses`.
    private final ByteBuffer zeroBatchClasses;
    // Where to store bottlenecks produced during inference.
    private ByteBuffer inferenceBottleneck;

    private ByteBuffer[] modelParameters;
    private final ByteBuffer[] modelGradients;
    private ByteBuffer[] nextModelParameters;
    private ByteBuffer[] optimizerState;
    private ByteBuffer[] nextOptimizerState;

    // 새로 추가할 데이터 및 targetName을 저장하기 위해서 만든 List
    private final List<TrainingSample> trainingSamples = new ArrayList<>();
    // 최소 1개의 코어를 사용하고 최대 사용가능한 코어수에서 1개를 제외한 모든 코어를 사용하겠다는 의미
    private static final int NUM_THREADS =
            Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
    // Used to spawn background threads.
    // 작업 처리에 사용되는 스레드를 제한된 개수만큼 정해 놓고 작업 Queue에 들어오는 작업들을 하나씩 스레드가 맡아 처리
    private final ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
    private final Lock trainingLock = new ReentrantLock();


    public TransferLearningModel(ModelLoader modelLoader, String[] classes){
        classesByIdx = classes;
        this.classes = new TreeMap<>();
        for (int classIdx = 0; classIdx < classes.length; classIdx++) {
            this.classes.put(classesByIdx[classIdx], classIdx);
        }

        try {
            initializeModel = new LiteInitializeModel(modelLoader.loadInitializeModel());   // model_init
//            inferenceModel = new LiteInferenceModel(modelLoader.loadInferenceModel(), numClasses);  // model_base
            inferenceModel = new LiteInferenceModel(modelLoader.loadInferenceModel(), 20);  // model_base
            bottleneckModel = new LiteBottleneckModel(modelLoader.loadBaseModel());     // model_head
            trainHeadModel = new LiteTrainHeadModel(modelLoader.loadTrainModel());      // model_train
            inferenceTrainModel = new LiteInferenceTrainModel(modelLoader.loadInferenceTrainModel(), numClasses);
            optimizerModel = new LiteOptimizerModel(modelLoader.loadOptimizerModel());      // optimizer
        } catch (IOException e) {
            throw new RuntimeException("Couldn't read underlying models for TransferLearningModel", e);
        }

        this.bottleneckShape = bottleneckModel.getBottleneckShape();
        int[] modelParameterSizes = trainHeadModel.getParameterSizes();

        modelParameters = new ByteBuffer[modelParameterSizes.length];
        modelGradients = new ByteBuffer[modelParameterSizes.length];
        nextModelParameters = new ByteBuffer[modelParameterSizes.length];

        for (int parameterIndex = 0; parameterIndex < modelParameterSizes.length; parameterIndex++) {
            int bufferSize = modelParameterSizes[parameterIndex] * FLOAT_BYTES;
            modelParameters[parameterIndex] = allocateBuffer(bufferSize);
            modelGradients[parameterIndex] = allocateBuffer(bufferSize);
            nextModelParameters[parameterIndex] = allocateBuffer(bufferSize);
        }

        initializeModel.initializeParameters(modelParameters);

        int[] optimizerStateElementSizes = optimizerModel.stateElementSizes();
        optimizerState = new ByteBuffer[optimizerStateElementSizes.length];
        nextOptimizerState = new ByteBuffer[optimizerStateElementSizes.length];

        for (int elemIdx = 0; elemIdx < optimizerState.length; elemIdx++) {
            int bufferSize = optimizerStateElementSizes[elemIdx] * FLOAT_BYTES;
            optimizerState[elemIdx] = allocateBuffer(bufferSize);
            nextOptimizerState[elemIdx] = allocateBuffer(bufferSize);
            fillBufferWithZeros(optimizerState[elemIdx]);
        }

        trainingBatchBottlenecks =
                allocateBuffer(getTrainBatchSize() * numBottleneckFeatures() * FLOAT_BYTES);
        int batchClassesNumElements = getTrainBatchSize() * numClasses;
        trainingBatchClasses = allocateBuffer(batchClassesNumElements * FLOAT_BYTES);
        zeroBatchClasses = allocateBuffer(batchClassesNumElements * FLOAT_BYTES);

        inferenceBottleneck = allocateBuffer(numBottleneckFeatures() * FLOAT_BYTES);
    }

    public Prediction[] predict(float[] input) {
        checkNotTerminating();
        inferenceLock.lock();

        try {
            if (isTerminating) {
                return null;
            }
            ByteBuffer inputBuffer = allocateBuffer(input.length * FLOAT_BYTES);
            for (float f : input) {
                inputBuffer.putFloat(f);
            }
            inputBuffer.rewind();

            float[] confidences;
            parameterLock.readLock().lock();
            try {
                confidences = inferenceModel.runInference(inputBuffer);
            } finally {
                parameterLock.readLock().unlock();
            }

            Prediction[] predictions = new Prediction[numClasses];
            for (int classIdx = 0; classIdx < numClasses; classIdx++) {
                predictions[classIdx] = new Prediction(classesByIdx[classIdx], confidences[classIdx]);
            }
            Arrays.sort(predictions, (a, b) -> -Float.compare(a.confidence, b.confidence));

            return predictions;
        } finally {
            inferenceLock.unlock();
        }
    }

    public Prediction[] predictTrain(float[] input, ByteBuffer[] trainParameters) {
        checkNotTerminating();
        inferenceLock.lock();

        try {
            if (isTerminating) {
                return  null;
            }
            ByteBuffer inputBuffer = allocateBuffer(input.length * FLOAT_BYTES);
            for (float f : input) {
                inputBuffer.putFloat(f);
            }
            inputBuffer.rewind();

            ByteBuffer bottleneck = bottleneckModel.generateBottleneck(inputBuffer, inferenceBottleneck);

            float[] confidences;
            parameterLock.readLock().lock();
            try {
                confidences = inferenceTrainModel.trainInference(bottleneck, trainParameters);
            } finally {
                parameterLock.readLock().unlock();
            }

            Prediction[] predictions = new Prediction[numClasses];
            for (int classIdx = 0; classIdx < numClasses; classIdx++) {
                predictions[classIdx] = new Prediction(classesByIdx[classIdx], confidences[classIdx]);
            }
            Arrays.sort(predictions, (a, b) -> -Float.compare(a.confidence, b.confidence));

            return predictions;
        } finally {
            inferenceLock.unlock();
        }
    }

    public Future<Void> addSample(ArrayList<ArrayList> data, ArrayList labels) {
        checkNotTerminating();

        for (int i=0; i<data.size(); i++) {
            if (!classes.containsKey(labels.get(i))) {
                throw new IllegalArgumentException(String.format(
                        "Class \"%s\" is not one of the classes recognized by the model", labels.get(i)));
            }
            ByteBuffer byteBuffer = allocateBuffer(data.get(i).size() * FLOAT_BYTES);

            for (int n=0; n<data.get(i).size(); n++) {
                float input = Float.parseFloat(data.get(i).get(n).toString());
                byteBuffer.putFloat(input);
            }
            byteBuffer.rewind();

            if (Thread.interrupted()) {   // Thread가 일시 정지 상태인 경우 return null
                return null;
            }
            // bottleneck 마지막 layer에 대한 feature를 output으로 하고 추가된 image와 model run 수행한 결과
            ByteBuffer bottleneck = bottleneckModel.generateBottleneck(byteBuffer, null);   // error

            trainingSamples.add(new TrainingSample(bottleneck, labels.get(i).toString()));

        }
        return null;
    }

    /**
     * Trains the model on the previously added data samples.
     *
     * @param numEpochs number of epochs to train for.
     * @param lossConsumer callback to receive loss values, may be null.
     * @return future that is resolved when training is finished.
     */
    public Future<Void> train(int numEpochs, LossConsumer lossConsumer) {   // Transfer Learning
        checkNotTerminating();
        if (trainingSamples.size() < getTrainBatchSize()) {
            throw new RuntimeException(
                    String.format(
                            "Too few samples to start training: need %d, got %d",
                            getTrainBatchSize(), trainingSamples.size()));
        }
        return executor.submit(() -> {
            trainingLock.lock();
            try {
                epochLoop:
                for (int epoch = 0; epoch < numEpochs; epoch++) {
                    float totalLoss = 0;
                    int numBatchesProcessed = 0;

                    for (List<TrainingSample> batch : trainingBatches()) {
                        if (Thread.interrupted()) {
                            break epochLoop;
                        }
                        trainingBatchClasses.put(zeroBatchClasses);
                        trainingBatchClasses.rewind();
                        zeroBatchClasses.rewind();

                        for (int sampleIdx = 0; sampleIdx < batch.size(); sampleIdx++) {
                            TrainingSample sample = batch.get(sampleIdx);
                            trainingBatchBottlenecks.put(sample.bottleneck);
                            sample.bottleneck.rewind();

                            // Fill trainingBatchClasses with one-hot.
                            int position =
                                    (sampleIdx * classes.size() + classes.get(sample.className)) * FLOAT_BYTES;
                            trainingBatchClasses.putFloat(position, 1);
                        }

                        trainingBatchBottlenecks.rewind();

                        float loss =
                                trainHeadModel.calculateGradients(
                                        trainingBatchBottlenecks,
                                        trainingBatchClasses,
                                        modelParameters,
                                        modelGradients);
                        totalLoss += loss;
                        numBatchesProcessed++;

                        optimizerModel.performStep(
                                modelParameters,
                                modelGradients,
                                optimizerState,
                                nextModelParameters,
                                nextOptimizerState);

                        ByteBuffer[] swapBufferArray;

                        // Swap optimizer state with its next version.
                        swapBufferArray = optimizerState;
                        optimizerState = nextOptimizerState;
                        nextOptimizerState = swapBufferArray;

                        // Swap model parameters with their next versions.
                        parameterLock.writeLock().lock();
                        try {
                            swapBufferArray = modelParameters;
                            modelParameters = nextModelParameters;
                            nextModelParameters = swapBufferArray;
                        } finally {
                            parameterLock.writeLock().unlock();
                        }
                    }

                    float avgLoss = totalLoss / numBatchesProcessed;
                    if (lossConsumer != null) {
                        lossConsumer.onLoss(epoch, avgLoss);
                    }
                }
                return null;
            } finally {
                trainingLock.unlock();
            }
        });
    }

    public interface LossConsumer {
        void onLoss(int epoch, float loss);
    }

    private static ByteBuffer allocateBuffer(int capacity) {
        // allocateDirect 같은 경우는 자바의 힙이 아닌 외부(운영체제 시스템)의 할당을 하게 됩니다.
        ByteBuffer buffer = ByteBuffer.allocateDirect(capacity);
        // buffer.order(ByteOrder.nativeOrder()) -> 기본적인 Native byte order로 설정
        buffer.order(ByteOrder.nativeOrder());
        return buffer;
    }

    private void checkNotTerminating() {
        if (isTerminating) {
            throw new IllegalStateException("Cannot operate on terminating model");
        }
    }

    private int numBottleneckFeatures() {
        int result = 1;
        for (int size : bottleneckShape) {
            result *= size;
        }

        return result;
    }

    /** Training model expected batch size. */
    public int getTrainBatchSize() {
        return trainHeadModel.getBatchSize();
    }

    private static class TrainingSample {
        ByteBuffer bottleneck;
        String className;

        TrainingSample(ByteBuffer bottleneck, String className) {
            this.bottleneck = bottleneck;
            this.className = className;
        }
    }

    /**
     * Constructs an iterator that iterates over training sample batches.
     * @return iterator over batches.
     */
    private Iterable<List<TrainingSample>> trainingBatches() {
        if (!trainingLock.tryLock()) {
            throw new RuntimeException("Thread calling trainingBatches() must hold the training lock");
        }
        trainingLock.unlock();

        Collections.shuffle(trainingSamples);
        return () ->
                new Iterator<List<TrainingSample>>() {
                    private int nextIndex = 0;

                    @Override
                    public boolean hasNext() {
                        return nextIndex < trainingSamples.size();
                    }

                    @Override
                    public List<TrainingSample> next() {
                        int fromIndex = nextIndex;
                        int toIndex = nextIndex + getTrainBatchSize();
                        nextIndex = toIndex;
                        if (toIndex >= trainingSamples.size()) {
                            // To keep batch size consistent, last batch may include some elements from the
                            // next-to-last batch.
                            return trainingSamples.subList(
                                    trainingSamples.size() - getTrainBatchSize(), trainingSamples.size());
                        } else {
                            return trainingSamples.subList(fromIndex, toIndex);
                        }
                    }
                };
    }

    /**
     * Prediction for a single class produced by the model.
     */
    public static class Prediction {
        private final String className;
        private final float confidence;

        public Prediction(String className, float confidence) {
            this.className = className;
            this.confidence = confidence;
        }

        public String getClassName() {
            return className;
        }

        public float getConfidence() {
            return confidence;
        }
    }

    private static void fillBufferWithZeros(ByteBuffer buffer) {
        int bufSize = buffer.capacity();
        int chunkSize = Math.min(1024, bufSize);

        ByteBuffer zerosChunk = allocateBuffer(chunkSize);
        for (int idx = 0; idx < chunkSize; idx++) {
            zerosChunk.put((byte) 0);
        }
        zerosChunk.rewind();

        for (int chunkIdx = 0; chunkIdx < bufSize / chunkSize; chunkIdx++) {
            buffer.put(zerosChunk);
        }
        for (int idx = 0; idx < bufSize % chunkSize; idx++) {
            buffer.put((byte) 0);
        }
    }

    public ByteBuffer[] saveModelParameters() {
        for (int i=0; i< modelParameters.length; i++) {
            modelParameters[i].rewind();
        }
        return modelParameters;
    }

    @Override
    public void close() throws IOException {
        isTerminating = true;
        inferenceModel.close();
    }
}
