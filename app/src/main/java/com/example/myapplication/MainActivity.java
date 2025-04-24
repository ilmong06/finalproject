package com.example.myapplication;

import java.util.Arrays;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.*;
import androidx.*;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

//Pytorch
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

//오디오 input
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;

//Android 입출력
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

//CHeck log
import android.util.Log;

//Connecting UI
import android.view.View;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;



public class MainActivity extends AppCompatActivity {

    private Module model;
    private TextView resultTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        resultTextView = findViewById(R.id.text_result);
        Button loadButton = findViewById(R.id.load_button);

        loadButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    String pa = assetFilePath(MainActivity.this, "wav2vec2.ptl");
                    File modelFile = new File(pa);
                    Log.d("Model", "Model PATH : "+pa);
                    Log.d("Model", "File Exist : "+modelFile.exists());
                    Log.d("Model", "File Length: "+modelFile.length()+" bytes");
                    // 모델 로딩
                    model = LiteModuleLoader.load(assetFilePath(MainActivity.this, "wav2vec2.ptl"));

                    // 예시 입력: 1초짜리 16kHz float 음성 텐서
                    float[] fakeAudio = new float[16000];
                    Tensor inputTensor = Tensor.fromBlob(fakeAudio, new long[]{1, 16000});

                    // 추론
                    Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
                    float[] output = outputTensor.getDataAsFloatArray();

                    // 결과 출력
                    resultTextView.setText("모델 출력: " + Arrays.toString(output));
                    Log.d("MyApp", "SUCCESS!");
                } catch (Exception e) {
                    resultTextView.setText("❌ 모델 로딩 실패: " + e.getMessage());
                    Log.e("MyApp", "FAILED." + e.getMessage());
                }
            }
        });




    }


    //For loading PyTorch Model File
    public static String assetFilePath(android.content.Context context, String assetName) throws java.io.IOException {
        java.io.File file = new java.io.File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (java.io.InputStream is = context.getAssets().open(assetName);
             java.io.FileOutputStream os = new java.io.FileOutputStream(file)) {
            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();
        }

        return file.getAbsolutePath();
    }


}