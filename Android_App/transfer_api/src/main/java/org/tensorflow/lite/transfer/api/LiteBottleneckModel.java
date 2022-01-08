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

import android.util.Log;

import java.io.Closeable;
import java.nio.ByteBuffer;

/**
 * A wrapper for TFLite model that generates bottlenecks from images.
 */
class LiteBottleneckModel implements Closeable {
  private static final String TAG = "TF_Lite";
  private static final int FLOAT_BYTES = 4;

  private final LiteModelWrapper modelWrapper;

  LiteBottleneckModel(LiteModelWrapper modelWrapper) {
    this.modelWrapper = modelWrapper;
  }

  /**
   * Passes a single image through the bottleneck model.
   * @param input image RGB data.
   * @param outBottleneck where to store the bottleneck. A new buffer is allocated if null.
   * @return bottleneck data. This is either [outBottleneck], or a newly allocated buffer.
   */
  synchronized ByteBuffer generateBottleneck(ByteBuffer input, ByteBuffer outBottleneck) {
    // modelWrapper.getInterpreter().getOutputTensor(0).numElements();
    if (outBottleneck == null) {
      outBottleneck = ByteBuffer.allocateDirect(getNumBottleneckFeatures() * FLOAT_BYTES);
    }
    // model run -> inference
    modelWrapper.getInterpreter().run(input, outBottleneck);    // error
    input.rewind();
    outBottleneck.rewind();

    return outBottleneck;
  }
  
  // outBottleneck이 null인 경우 사용
  int getNumBottleneckFeatures() {
    // 마지막 layer에 대한 output feature 개수 반환(7 x 7 x 1280) = 62720
    return modelWrapper.getInterpreter().getOutputTensor(0).numElements();  // error
  }

  int[] getBottleneckShape() {
    // getInterpreter -> Interpreter interpreter 선언
    // interpreter.getOutputTensor(int outputIndex) -> Tensor 모양 및 유형 정보, outputIndex
    // outputindex : 0 -> 마지막층 -> mobileNet v2 기준 마지막 conv 연산 후 output에 대한 feature 의미
    // (7, 7, 1280)
    return modelWrapper.getInterpreter().getOutputTensor(0).shape();
  }

  @Override
  public void close() {
    modelWrapper.close();
  }
}
