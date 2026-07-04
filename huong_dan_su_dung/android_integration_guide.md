# Hướng Dẫn Tích Hợp Fall Detection Lên Android (Kotlin & TFLite)

Tài liệu này hướng dẫn bạn cách tích hợp pipeline Fall Detection (YOLO-Pose + LSTM) lên Android bằng ngôn ngữ Kotlin, sử dụng công nghệ **TensorFlow Lite (TFLite)**. Mô hình đã được ép cứng kích thước 480x640 (chuẩn dọc Mobile) và tích hợp thuật toán "Bộ lọc rơi" (Drop Velocity) cùng với "Duy trì cảnh báo ngã 3 giây" (State Sustain).

---

## 1. Chuẩn Bị Mô Hình (TFLite Models)

Hãy copy 2 file TFLite sau vào thư mục `app/src/main/assets/` của project Android Studio của bạn:
1. `yolo26n-pose.tflite` (hoặc bản nhẹ `yolo26n-pose_w8a16.tflite`)
2. `fall_lstm_v2_fp32.tflite`

---

## 2. Cài Đặt Thư Viện (build.gradle)

Bạn cần thêm các dependency sau vào file `app/build.gradle`:

```gradle
dependencies {
    // CameraX để lấy luồng video từ camera điện thoại
    def camerax_version = "1.3.0"
    implementation "androidx.camera:camera-core:${camerax_version}"
    implementation "androidx.camera:camera-camera2:${camerax_version}"
    implementation "androidx.camera:camera-lifecycle:${camerax_version}"
    implementation "androidx.camera:camera-view:${camerax_version}"

    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

---

## 3. Kiến Trúc Pipeline Bằng Kotlin

### A. Quản Lý Trạng Thái Ngã (State Maintainer)
Theo thuật toán mới nhất, ta cần duy trì cảnh báo "FALL" trong 3 giây (tương đương 90 frames nếu video là 30 FPS).
Tạo một biến toàn cục hoặc property trong class Camera của bạn:

```kotlin
// Key: trackId, Value: số frame còn lại để đếm ngược (ví dụ 90 -> 0)
private val fallTimers = mutableMapOf<Int, Int>()
```

### B. Lớp `FallPipeline` (TFLite)
Lớp này đóng vai trò load 2 mô hình TFLite và xử lý luồng dữ liệu 30 frames.

```kotlin
import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.FloatBuffer

class FallPipeline(context: Context) {
    private var yoloInterpreter: Interpreter
    private var lstmInterpreter: Interpreter

    // Lưu trữ buffer 30 frames cho TỪNG ID người (SEQ_LEN = 30)
    private val trackHistory = mutableMapOf<Int, MutableList<FloatArray>>()

    init {
        // Load YOLO-Pose
        val yoloModel = FileUtil.loadMappedFile(context, "yolo26n-pose.tflite")
        val yoloOptions = Interpreter.Options().apply { setNumThreads(4) }
        yoloInterpreter = Interpreter(yoloModel, yoloOptions)

        // Load LSTM
        val lstmModel = FileUtil.loadMappedFile(context, "fall_lstm_v2_fp32.tflite")
        val lstmOptions = Interpreter.Options().apply { setNumThreads(2) }
        lstmInterpreter = Interpreter(lstmModel, lstmOptions)
    }

    /**
     * Hàm gọi mỗi frame cho mỗi người (trackId). 
     * Trả về Pair<Boolean, Boolean> -> (lstm_says_fall, hip_actually_dropped)
     */
    fun processPerson(trackId: Int, keypoints: Array<FloatArray>): Pair<Boolean, Boolean> {
        val buffer = trackHistory.getOrPut(trackId) { mutableListOf() }
        
        // keypoints là mảng 17 điểm, mỗi điểm chứa [x, y, conf]
        buffer.add(flattenKeypoints(keypoints))

        if (buffer.size > 30) buffer.removeAt(0)
        
        if (buffer.size == 30) {
            // 1. Chuẩn hóa chuỗi 30 frames thành mảng (30 x 38 features)
            val normalizedData = normalizeWindow(buffer) 
            
            // 2. Chạy LSTM (Input: [1, 30, 38])
            val inputBuffer = FloatBuffer.wrap(normalizedData)
            val outputBuffer = FloatBuffer.allocate(1) // Output: [1, 1]
            
            lstmInterpreter.run(inputBuffer, outputBuffer)
            
            val prob = outputBuffer.get(0)
            
            // 3. Tính toán độ rơi của hông (Velocity Drop Filter)
            val dropRatio = computeDropRatio(buffer)
            
            val lstmSaysFall = prob > 0.5f
            val hipActuallyDropped = dropRatio > 0.15f
            
            return Pair(lstmSaysFall, hipActuallyDropped)
        }
        
        return Pair(false, false)
    }

    private fun flattenKeypoints(kpts: Array<FloatArray>): FloatArray {
        // Chuyển cấu trúc 2D thành mảng 1D (51 phần tử) để dễ lưu trữ
        val flat = FloatArray(17 * 3)
        for (i in 0 until 17) {
            flat[i * 3] = kpts[i][0]
            flat[i * 3 + 1] = kpts[i][1]
            flat[i * 3 + 2] = kpts[i][2] // conf
        }
        return flat
    }

    private fun normalizeWindow(buffer: List<FloatArray>): FloatArray {
        // Bạn cần port hàm normalize_window() từ Python sang đây.
        // Kết quả trả về phải là một mảng FloatArray có kích thước 30 * 38 = 1140 phần tử
        return FloatArray(30 * 38)
    }

    private fun computeDropRatio(buffer: List<FloatArray>): Float {
        // Bạn cần port hàm compute_drop_ratio() từ Python sang đây.
        // Trả về số thực đại diện cho drop_ratio (ví dụ 0.2f)
        return 0f
    }
}
```

---

## 4. Tích Hợp Lên CameraX & Tracking (Vòng lặp chính)

Trong class phân tích CameraX (`ImageAnalysis.Analyzer`), hãy cập nhật logic duy trì trạng thái ngã 3 giây (90 frames):

```kotlin
imageAnalysis.setAnalyzer(executor) { imageProxy ->
    val bitmap = imageProxy.toBitmap() // Nhớ resize về 480x640 bằng Letterbox
    
    // 1. Chạy YOLO và gán ID (Tracking)
    // val trackedPeople = ... 
    
    // 2. Duyệt qua từng người
    for (person in trackedPeople) {
        val trackId = person.id
        val (lstmSaysFall, hipDropped) = pipeline.processPerson(trackId, person.keypoints)
        
        // 3. Thuật toán duy trì cảnh báo
        if (lstmSaysFall && hipDropped) {
            // Ngã thật -> Set timer đếm ngược 90 frames (khoảng 3 giây ở 30 FPS)
            fallTimers[trackId] = 90
        }
        
        // 4. Quyết định màu khung và hiển thị
        val currentTimer = fallTimers.getOrDefault(trackId, 0)
        
        if (currentTimer > 0) {
            fallTimers[trackId] = currentTimer - 1
            // TRẠNG THÁI NGÃ (Duy trì)
            drawBox(person.box, Color.RED, "FALL DETECTED!")
            
            // Nếu muốn phát chuông cảnh báo, hãy kích hoạt tại đây.
        } else if (lstmSaysFall && !hipDropped) {
            // LSTM báo ngã nhưng hông không di chuyển -> Đang nằm
            drawBox(person.box, Color.YELLOW, "LYING")
        } else {
            // Bình thường
            drawBox(person.box, Color.GREEN, "OK")
        }
    }
    
    imageProxy.close()
}
```

## Chú ý quan trọng:
Khâu vất vả nhất trên Android sẽ là 2 hàm `normalizeWindow` và `computeDropRatio`. Bạn phải đảm bảo thứ tự của 17 điểm khớp xương, cách tính tỷ lệ `aspect_ratio`, công thức tính vận tốc đầu, hông phải khớp **100% từng con số** với bản Python. Nếu không, mô hình LSTM sẽ nhận dữ liệu bị nhiễu và dự đoán sai.
