/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.transfer.api;

import java.io.Closeable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.TreeMap;

class LiteInferenceTrainModel implements Closeable {
  private static final int FLOAT_BYTES = 4;

  private final LiteModelWrapper modelWrapper;
  private final int numClasses;

  LiteInferenceTrainModel(LiteModelWrapper modelWrapper, int numClasses) {
    this.modelWrapper = modelWrapper;
    this.numClasses = numClasses;
  }

  float[] trainInference(ByteBuffer bottleneck, ByteBuffer[] modelParameters) {
    // predictionsBuffer -> 최종 target count만큼 buffer 생성
    ByteBuffer predictionsBuffer = ByteBuffer.allocateDirect(numClasses * FLOAT_BYTES);
    predictionsBuffer.order(ByteOrder.nativeOrder());

    // TreeMap -> 키와 값이 저장된 Map, Etnry로 구성
    // 객체를 저장하면 자동으로 정렬되는데, 키는 저장과 동시에 자동 오름차순으로 정렬
    Map<Integer, Object> outputs = new TreeMap<>();
    // key : 0, value : predictionsBuffer
    outputs.put(0, predictionsBuffer);

    // input을 Object Array로 선언
    Object[] inputs = new Object[modelParameters.length + 1];
    inputs[0] = bottleneck;
    // modelParameters 데이터를 inputs로 복사
    System.arraycopy(modelParameters, 0, inputs, 1, modelParameters.length);

    // input과 output이 각 1개인 경우 run() / 여러개인 경우 runForMultipleInputsOutputs() 사용
    modelWrapper.getInterpreter().runForMultipleInputsOutputs(inputs, outputs);

    // bottleneck, modelParameters, predictionsBuffer 가리키는 곳 0으로 이동
    bottleneck.rewind();
    for (ByteBuffer buffer : modelParameters) {
      buffer.rewind();
    }
    predictionsBuffer.rewind();

    // predictionsBuffer에 결과 값이 저장되어 있음
    // predictions array에 결과 값 저장
    float[] predictions = new float[numClasses];
    for (int classIdx = 0; classIdx < numClasses; classIdx++) {
      predictions[classIdx] = predictionsBuffer.getFloat();
    }

    return predictions;
  }

  @Override
  public void close() {
    modelWrapper.close();
  }
}
