# Hướng Dẫn Tích Hợp Fall Detection Lên Android (Kotlin)

Quá trình đưa hệ thống từ Python (Mac) lên Android cần một số chuyển đổi về mặt công cụ. 
Lúc đầu chúng ta định dùng **TFLite**, tuy nhiên việc convert sang TFLite trên môi trường Python 3.14 hiện tại của bạn đang bị lỗi thư viện từ phía Google (`litert-torch`). Ngoài ra, mô hình LSTM bằng PyTorch lại cực kỳ khó chuyển sang TFLite do đặc thù của kiến trúc chuỗi.

Do đó, **Giải pháp tối ưu nhất cho Android (chuẩn công nghiệp)** là sử dụng **ONNX Runtime Mobile** cho **CẢ 2 MÔ HÌNH (YOLO-Pose và LSTM)**. ONNX Runtime do Microsoft phát triển, hỗ trợ Android cực kỳ tốt và chạy trực tiếp file `.onnx` từ PyTorch mà không lo mất mát dữ liệu hay sai lệch cấu trúc!

Dưới đây là hướng dẫn các bước thực hiện bằng Kotlin.

---

## 1. Xuất Mô Hình (Model Export)

Mình đã chạy lệnh xuất YOLO sang ONNX cho bạn (nó sẽ tạo ra file `yolo26n-pose.onnx`):
```bash
./venv/bin/yolo export model=yolo26n-pose.pt format=onnx
```

Đối với LSTM, bạn hãy sử dụng file `fall_lstm_best.onnx` đã có sẵn. 
👉 Hãy copy 2 file `.onnx` này vào thư mục `app/src/main/assets/` của project Android Studio của bạn.

---

## 2. Cài Đặt Thư Viện (build.gradle)

Bạn cần thêm các dependency sau vào `app/build.gradle`:

```gradle
dependencies {
    // CameraX để lấy luồng video
    def camerax_version = "1.3.0"
    implementation "androidx.camera:camera-core:${camerax_version}"
    implementation "androidx.camera:camera-camera2:${camerax_version}"
    implementation "androidx.camera:camera-lifecycle:${camerax_version}"
    implementation "androidx.camera:camera-view:${camerax_version}"

    // ONNX Runtime cho cả YOLO và LSTM
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.1'
}
```

---

## 3. Kiến Trúc Pipeline Trong Kotlin

Bạn sẽ cần tạo 3 module chính trong Kotlin:

### A. Lớp `YoloPoseDetector` (ONNX)
Lớp này nhận vào ảnh `Bitmap`, resize về 640x640, convert sang mảng FloatArray và chạy qua ONNX Runtime. Output sẽ là tensor `[1, 57, 8400]`.
Bạn sẽ cần viết một hàm NMS (Non-Maximum Suppression) trong Kotlin để lọc ra Bounding Box và Keypoints.

### B. Lớp `ByteTracker` (Tracking Đa Mục Tiêu)
Trong Python chúng ta gọi `yolo.track()`. Trên Android, bạn cần implement một thuật toán Tracking (ByteTrack hoặc SORT) bằng Kotlin. Nếu muốn đơn giản, bạn có thể tự viết hàm **IoU Tracker** (Intersection over Union) như sau:
```kotlin
// Hàm so sánh Box hiện tại với các Box ở Frame trước bằng IoU để gán ID
fun matchTracking(currentBoxes: List<RectF>, previousTracks: Map<Int, RectF>): Map<Int, RectF> {
    // Logic gán ID cho người mới, giữ ID cho người cũ bằng cách tìm IoU cao nhất
}
```

### C. Lớp `LstmFallDetector` (ONNX Runtime)
Lớp này sẽ tái tạo lại chính xác hàm `normalize_window` của Python và chạy mô hình ONNX LSTM.

```kotlin
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

class LstmFallDetector(assetManager: AssetManager) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // Lưu trữ buffer 15 frames cho TỪNG ID
    private val trackHistory = mutableMapOf<Int, MutableList<FloatArray>>()

    init {
        val modelBytes = assetManager.open("fall_lstm_best.onnx").readBytes()
        session = env.createSession(modelBytes, OrtSession.SessionOptions())
    }

    fun processPerson(trackId: Int, keypoints: Array<FloatArray>, bboxW: Float, bboxH: Float, frameHeight: Float): Float {
        val buffer = trackHistory.getOrPut(trackId) { mutableListOf() }
        
        // Flatten 17 keypoints (3 trục X, Y, Conf) thành mảng 1D
        val flatKpts = FloatArray(17 * 3)
        // ... (copy data)
        buffer.add(flatKpts)

        if (buffer.size > 15) buffer.removeAt(0)
        
        if (buffer.size == 15) {
            // Thực hiện Normalize giống hệt hàm normalize_window trong Python
            val normalizedData = normalizeWindow(buffer) 

            // Chạy LSTM
            val tensorShape = longArrayOf(1, 15, 37)
            val floatBuffer = FloatBuffer.wrap(normalizedData)
            val inputTensor = OnnxTensor.createTensor(env, floatBuffer, tensorShape)
            
            val result = session.run(mapOf("input" to inputTensor))
            val outputTensor = result.get(0).value as Array<FloatArray>
            
            val prob = sigmoid(outputTensor[0][0])
            
            // Các bộ lọc nhiễu chuẩn hóa (Kích thước và Aspect Ratio)
            val aspectRatio = bboxW / bboxH
            if (prob > 0.5f && aspectRatio > 0.6f && bboxH > (frameHeight / 3.0f)) {
                return prob // Xác nhận té ngã
            }
        }
        return 0f
    }

    private fun normalizeWindow(buffer: List<FloatArray>): FloatArray {
        // Implement logic: 
        // 1. Dịch tâm hông (Pelvis) về (0,0)
        // 2. Scale bằng max(bboxW, bboxH)
        // 3. Tính Delta_X, Delta_Y (Vận tốc)
        // Lưu ý: Code này phải giống 100% logic trong file data_preparation.py
        return FloatArray(15 * 37) 
    }

    private fun sigmoid(x: Float): Float {
        return (1 / (1 + Math.exp(-x.toDouble()))).toFloat()
    }
}
```

---

## 4. Tích Hợp Lên CameraX (Vòng lặp chính)

```kotlin
imageAnalysis.setAnalyzer(executor) { imageProxy ->
    val bitmap = imageProxy.toBitmap()
    
    // 1. Chạy YOLO
    val yoloResults = yoloDetector.detect(bitmap)
    
    // 2. Tracking ID
    val trackedPeople = byteTracker.update(yoloResults)
    
    // 3. Duyệt qua từng người
    for (person in trackedPeople) {
        val prob = lstmDetector.processPerson(person.id, person.keypoints, person.width, person.height, bitmap.height.toFloat())
        
        if (prob > 0.5f) {
            // Báo động rớt: Vẽ khung Đỏ
            drawBox(person.box, Color.RED, "FALL DETECTED")
        } else {
            // Bình thường: Vẽ khung Xanh
            drawBox(person.box, Color.GREEN, "NORMAL")
        }
    }
    
    imageProxy.close()
}
```

## Chú ý quan trọng:
Khâu vất vả nhất trên Android sẽ là hàm `normalizeWindow`. Bạn phải đảm bảo thứ tự của 17 điểm khớp xương, cách tính tỷ lệ, và công thức `delta_x = (cx - prev_cx) / prev_scale` phải khớp 100% từng con số với bản Python, nếu không mô hình LSTM sẽ hoạt động không chính xác.
