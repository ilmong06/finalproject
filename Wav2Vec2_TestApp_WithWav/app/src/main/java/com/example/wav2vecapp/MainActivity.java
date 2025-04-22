package com.example.wav2vecapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.*;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.*;

public class MainActivity extends AppCompatActivity {

    private MediaRecorder recorder;
    private String filePath;
    private TextView textView;
    private Button startButton, stopButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textView = findViewById(R.id.textResult);
        startButton = findViewById(R.id.startButton);
        stopButton = findViewById(R.id.stopButton);

        filePath = getExternalFilesDir(null).getAbsolutePath() + "/recorded.wav";

        startButton.setOnClickListener(view -> {
            // 테스트용 파일 서버 전송
            sendTestWavFromAssets();
        });

        stopButton.setOnClickListener(view -> stopRecording());

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            }, 200);
        }
    }

    private void startRecording() {
        recorder = new MediaRecorder();
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
        recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
        recorder.setOutputFile(filePath);

        try {
            recorder.prepare();
            recorder.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void stopRecording() {
        try {
            recorder.stop();
            recorder.release();
            recorder = null;

            // 기존 방식 사용 시
            // sendAudioFileToServer(new File(filePath));

            // 테스트용 .wav 파일을 서버로 보내기
            sendTestWavFromAssets();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // ✅ 테스트용 test.wav 파일 전송 함수
    private void sendTestWavFromAssets() {
        try {
            InputStream inputStream = getAssets().open("test.wav");
            File outFile = new File(getCacheDir(), "test.wav");

            OutputStream outputStream = new FileOutputStream(outFile);
            byte[] buffer = new byte[1024];
            int length;
            while ((length = inputStream.read(buffer)) > 0) {
                outputStream.write(buffer, 0, length);
            }
            outputStream.close();
            inputStream.close();

            sendAudioFileToServer(outFile);

        } catch (IOException e) {
            e.printStackTrace();
            textView.setText("파일 전송 오류: " + e.getMessage());
        }
    }

    // ✅ 서버로 파일 전송 함수
    private void sendAudioFileToServer(File file) {
        RequestBody reqFile = RequestBody.create(MediaType.parse("audio/wav"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", file.getName(), reqFile);

        HttpLoggingInterceptor logging = new HttpLoggingInterceptor();
        logging.setLevel(HttpLoggingInterceptor.Level.BODY);

        OkHttpClient client = new OkHttpClient.Builder()
                .addInterceptor(logging)
                .build();

        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("http://10.0.2.2:5000/")  // Flask 서버 IP
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build();

        ApiService apiService = retrofit.create(ApiService.class);
        Call<TranscriptionResponse> call = apiService.uploadAudio(body);

        call.enqueue(new Callback<TranscriptionResponse>() {
            @Override
            public void onResponse(Call<TranscriptionResponse> call, Response<TranscriptionResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    textView.setText("결과: " + response.body().text);
                } else {
                    textView.setText("서버 응답 실패");
                }
            }

            @Override
            public void onFailure(Call<TranscriptionResponse> call, Throwable t) {
                textView.setText("에러: " + t.getMessage());
            }
        });
    }
}
