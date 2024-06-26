package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.*

internal data class FaceDetectorResult (
    var outputBitmap: Bitmap,
    var outputBox: Array<FloatArray>
) {}

internal class FaceDetector() {

    fun detect(inputStream: InputStream, ortEnv: OrtEnvironment, ortSession: OrtSession, imgBitmap: Bitmap): FaceDetectorResult {

        var imageBitmat =  BitmapFactory.decodeStream(inputStream)
        val preprocessedImage = preprocessImage(imageBitmat)

        // Convert FloatArray to FloatBuffer
        val floatBuffer = FloatBuffer.wrap(preprocessedImage)

        // Run inference
        val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, longArrayOf(1, 3, 640, 640))

        //
        inputTensor.use {
            val results = ortSession.run(mapOf("input" to inputTensor))
            val output = results[0].value as Array<Array<FloatArray>>

            // Post-process the output
            val detections = postProcess(output)
            println("Detections ${detections.size}")
//            detections.forEach { detection ->
//                println("Detection: ${detection.label}, Confidence: ${detection.confidence}, Box: ${detection.box}")
//            }

//            val output = results[0].value as Array<FloatArray>

            // Post-process the output
//            val detections = postProcess(output)
//            detections.forEach { detection ->
//                println("Detection: ${detection.label}, Confidence: ${detection.confidence}, Box: ${detection.box}")
//            }
//            val output = ortSession.run(
//                Collections.singletonMap("input", inputTensor),
//                setOf("output")
//            )
//
//            output.use {
//                val rawOutput1 = (output?.get(0)?.value) as Array<FloatArray>
//
//                Log.d("TAG", "Item")
//
//                rawOutput1.forEach {
//                    Log.d("TAG", "Item ${it.joinToString(separator = ",")}")
//                }

                return FaceDetectorResult(imgBitmap, output.get(0))
//            }
        }
    }

    // Preprocess the image: resize, normalize, and convert to float array
    // Preprocess the image: resize, normalize, and convert to float array
    fun preprocessImage(image: Bitmap): FloatArray {
        val resizedImage = Bitmap.createScaledBitmap(image, 640, 640, true)
        val floatArray = FloatArray(1 * 3 * 640 * 640)
        val buffer = IntArray(640 * 640)
        resizedImage.getPixels(buffer, 0, 640, 0, 0, 640, 640)

        for (i in buffer.indices) {
            val pixel = buffer[i]
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            floatArray[i] = r
            floatArray[640 * 640 + i] = g
            floatArray[2 * 640 * 640 + i] = b
        }
        return floatArray
    }

    // Post-process the output to extract bounding boxes and confidence scores
    data class Detection(val label: String, val confidence: Float, val box: FloatArray)

    fun postProcess(output: Array<Array<FloatArray>>): List<Detection> {
        val detections = mutableListOf<Detection>()
        output[0].forEach { detection ->
            val confidence = detection[4]
            if (confidence > 0.5) {
                val box = detection.sliceArray(0..3)
                val label = "face"
                detections.add(Detection(label, confidence, box))
            }
        }
        return detections
    }




    private fun byteArrayToBitmap(data: ByteArray): Bitmap {
        return BitmapFactory.decodeByteArray(data, 0, data.size)
    }
}