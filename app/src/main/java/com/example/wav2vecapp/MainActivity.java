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

import org.json.JSONObject;

import java.io.File;
import java.util.List;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class MainActivity extends AppCompatActivity {

    private MediaRecorder recorder;
    private String filePath;
    private TextView textView, textRegisterStep;
    private Button startButton, stopButton, registerButton;
    private boolean isRegistering = false;
    private int registerCount = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textView = findViewById(R.id.textResult);
        textRegisterStep = findViewById(R.id.textRegisterStep);
        startButton = findViewById(R.id.startButton);
        stopButton = findViewById(R.id.stopButton);
        registerButton = findViewById(R.id.registerButton);

        filePath = getExternalFilesDir(null).getAbsolutePath() + "/recorded.wav";

        startButton.setOnClickListener(view -> {
            isRegistering = false;
            startRecording();
        });

        stopButton.setOnClickListener(view -> stopRecording());

        registerButton.setOnClickListener(view -> {
            isRegistering = true;
            registerCount = 0;
            textRegisterStep.setText("🧬 1/4 회차 등록 시작");
            startRecording();
        });

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            }, 200);
        }
    }

    private void startRecording() {
        try {
            recorder = new MediaRecorder();
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
            recorder.setOutputFile(filePath);

            recorder.prepare();
            recorder.start();

            textView.setText("🎙️ 녹음 중... (Tap 종료)");
            stopButton.setEnabled(true);
            startButton.setEnabled(false);
            registerButton.setEnabled(false);

        } catch (Exception e) {
            e.printStackTrace();
            textView.setText("❌ 녹음 시작 실패");
        }
    }

    private void stopRecording() {
        try {
            recorder.stop();
            recorder.release();
            recorder = null;

            textView.setText("🎧 녹음 종료됨");

            if (isRegistering) {
                sendAudioToRegister(filePath);
            } else {
                sendAudioToServer(filePath);
            }

            stopButton.setEnabled(false);
            startButton.setEnabled(true);
            registerButton.setEnabled(true);

        } catch (Exception e) {
            e.printStackTrace();
            textView.setText("❌ 녹음 중지 실패");
        }
    }

    private void sendAudioToServer(String filePath) {
        File file = new File(filePath);
        RequestBody reqFile = RequestBody.create(MediaType.parse("audio/wav"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", file.getName(), reqFile);

        Retrofit retrofit = getRetrofitClient();
        ApiService apiService = retrofit.create(ApiService.class);
        Call<TranscriptionResponse> call = apiService.uploadAudio(body);

        call.enqueue(new Callback<TranscriptionResponse>() {
            @Override
            public void onResponse(Call<TranscriptionResponse> call, Response<TranscriptionResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    StringBuilder sb = new StringBuilder();
                    sb.append("📝 텍스트 결과:\n").append(response.body().text).append("\n\n");

                    List<Float> speakerVector = response.body().speakerVector;
                    if (speakerVector != null) {
                        sb.append("🧬 화자 벡터:\n");
                        for (Float val : speakerVector) {
                            sb.append(String.format("%.4f ", val));
                        }
                    }
                    textView.setText(sb.toString());
                } else {
                    textView.setText("❌ 서버 오류 또는 화자 인증 실패");
                }
            }

            @Override
            public void onFailure(Call<TranscriptionResponse> call, Throwable t) {
                textView.setText("🚫 연결 실패: " + t.getMessage());
            }
        });
    }

    private void sendAudioToRegister(String filePath) {
        File file = new File(filePath);
        RequestBody reqFile = RequestBody.create(MediaType.parse("audio/wav"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", file.getName(), reqFile);

        Retrofit retrofit = getRetrofitClient();
        ApiService apiService = retrofit.create(ApiService.class);

        Call<ResponseBody> call = apiService.registerSpeaker(body);
        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                if (response.isSuccessful()) {
                    try {
                        String responseBody = response.body().string();
                        String msg = new JSONObject(responseBody).getString("message");

                        textView.setText("✅ " + msg);
                        textRegisterStep.setText(msg);

                        if (msg.contains("4/4")) {
                            // ✅ 등록 완료. 반복 종료!
                            isRegistering = false;
                            registerCount = 0;
                            stopButton.setEnabled(true);
                            startButton.setEnabled(true);
                            registerButton.setEnabled(true);
                        } else {
                            // ✅ 등록 진행 중이면 다음 회차로
                            registerCount++;

                            // ⛔ 4회차 초과 방지
                            if (registerCount < 4) {
                                textView.setText("🎤 " + (registerCount + 1) + "/4 회차 녹음 시작");
                                startRecording();
                            }
                        }

                    } catch (Exception e) {
                        textView.setText("⚠️ 응답 파싱 오류");
                        e.printStackTrace();
                    }
                } else {
                    textView.setText("❌ 등록 실패 (서버 오류)");
                }
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                textView.setText("🚫 등록 요청 실패: " + t.getMessage());
            }
        });
    }


    private Retrofit getRetrofitClient() {
        HttpLoggingInterceptor logging = new HttpLoggingInterceptor();
        logging.setLevel(HttpLoggingInterceptor.Level.BODY);
        OkHttpClient client = new OkHttpClient.Builder()
                .addInterceptor(logging)
                .build();

        return new Retrofit.Builder()
                .baseUrl(BuildConfig.FLASK_BASE_URL)
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
    }
}