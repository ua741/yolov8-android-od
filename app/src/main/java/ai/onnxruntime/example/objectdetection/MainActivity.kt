package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.*
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import java.io.InputStream
import java.util.*

class MainActivity : AppCompatActivity() {
    private var faceOrtEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var faceOrtSession: OrtSession
    private lateinit var inputImage: ImageView
    private lateinit var outputImage: ImageView
    private lateinit var objectDetectionButton: Button
    private var imageid = 0
    private lateinit var classes: List<String>

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Log.i(TAG, "onCreate: Model read complete")
        inputImage = findViewById(R.id.imageView1)
        outputImage = findViewById(R.id.imageView2)
        objectDetectionButton = findViewById(R.id.object_detection_button)
        inputImage.setImageBitmap(
            BitmapFactory.decodeStream(readInputImage())
        )
        imageid = 0


        val sessionOptions2: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions2.setInterOpNumThreads(1)
        sessionOptions2.setIntraOpNumThreads(1)
        sessionOptions2.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
//        sessionOptions2.
//        sessionOptions2.enableProfiling()

        faceOrtSession = faceOrtEnv.createSession(readFaceDeteionModel(), sessionOptions2)

        objectDetectionButton.setOnClickListener {
            try {
                val numberOfDetections = 10000
                var elapsedTime = 0.toLong()
                Coroutines.io {
                    val startTime = System.currentTimeMillis()
                    var i = 1;
                    while (i < numberOfDetections) {
                        performFaceDetection(faceOrtSession, i)
                        i += 1;
                        performFaceDetection(faceOrtSession, i)
                        i += 1;
                        performFaceDetection(faceOrtSession, i)
                        i += 1;
                        performFaceDetection(faceOrtSession, i)
                        i += 1;
                        performFaceDetection(faceOrtSession, i)
                        i += 1;
                        performFaceDetection(faceOrtSession, i)

                    }

                    val endTime = System.currentTimeMillis()
                    elapsedTime = endTime - startTime
                    val fps = 1000.0 / elapsedTime

                    Log.i(
                        TAG,
                        "Perform Detection Loop Total $numberOfDetections detections in elapsedTime : $elapsedTime, FPS: $fps"
                    )
                    runOnUiThread {
                        Toast.makeText(
                            baseContext,
                            "ObjectDetection performed! $numberOfDetections in $elapsedTime,",
                            Toast.LENGTH_SHORT
                        )
                            .show()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Exception caught when performing ObjectDetection", e)
                runOnUiThread {
                    Toast.makeText(
                        baseContext,
                        "Failed to perform ObjectDetection",
                        Toast.LENGTH_SHORT
                    )
                        .show()
                }
            }
        }
        Log.i(TAG, "onCreate: Initialization complete")
    }

    override fun onDestroy() {
        super.onDestroy()
        faceOrtEnv.close()
        faceOrtSession.close()
        Log.i(TAG, "onDestroy: Resources released")
    }

//    private fun updateUI(result: Result, elapsedTime: Long, fps: Double) {
//        val mutableBitmap: Bitmap = result.outputBitmap.copy(Bitmap.Config.ARGB_8888, true)
//        val canvas = Canvas(mutableBitmap)
//        val paint = Paint().apply {
//            color = Color.RED
//            textSize = 28f
//            xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER)
//        }
//
//        canvas.drawBitmap(mutableBitmap, 0.0f, 0.0f, paint)
//        result.outputBox.forEach { boxInfo ->
//            canvas.drawText(
//                "%s:%.2f".format(classes[boxInfo[5].toInt()], boxInfo[4]),
//                boxInfo[0] - boxInfo[2] / 2,
//                boxInfo[1] - boxInfo[3] / 2,
//                paint.apply { style = Paint.Style.FILL }
//            )
//            val left = boxInfo[0] - boxInfo[2] / 2
//            val top = boxInfo[1] - boxInfo[3] / 2
//            val right = boxInfo[0] + boxInfo[2] / 2
//            val bottom = boxInfo[1] + boxInfo[3] / 2
//
//            canvas.drawRect(left, top, right, bottom, paint.apply {
//                style = Paint.Style.STROKE
//                strokeWidth = 2.0f
//            })
//        }
//
//        Coroutines.main {
//            outputImage.setImageBitmap(mutableBitmap)
//        }
//
//        Log.d(TAG, "PerformanceOut: ElapsedTime: $elapsedTime, FPS: $fps")
//        canvas.drawText("ElapsedTime: $elapsedTime, FPS: $fps", 0.0f, 50.0f, paint.apply {
//            color = Color.WHITE
//            style = Paint.Style.FILL
//            textSize = 20.0f
//            setShadowLayer(5.0f, 0.0f, 0.0f, Color.BLACK)
//        })
//    }


    private fun readFaceDeteionModel(): ByteArray {
        val modelID = R.raw.yolov5s_face_640_640_dynamic
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readInputImage(): InputStream {
        imageid = Random().nextInt(1)
        return assets.open("test_object_detection_1.jpg")
    }

    private fun performFaceDetection(faceOrtSession: OrtSession, loopIndex: Int) {
        var faceDetector = FaceDetector()

        val startTime = System.currentTimeMillis()
        val imageStream = readInputImage()
        val imgBitmap = BitmapFactory.decodeStream(imageStream)

        Coroutines.main {
            inputImage.setImageBitmap(imgBitmap)
        }

        imageStream.reset()
        val result = faceDetector.detect(imageStream, faceOrtEnv, faceOrtSession, imgBitmap)
        val endTime = System.currentTimeMillis()

        val elapsedTime = endTime - startTime
        val fps = 1000.0 / elapsedTime

//        updateUI(result, elapsedTime, fps)

        Log.d(TAG, "Index $loopIndex: Total elapsedTime : $elapsedTime, fps : $fps")
    }

    companion object {
        const val TAG = "ORTObjectDetection"
    }
}
